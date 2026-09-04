"""جمع کردن شمارنده های پست های عمومی تلگرام، بدون ذخیره متن پست."""

import argparse
import asyncio
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from telethon import TelegramClient, functions
from telethon.errors import FloodWaitError

try:
    from telegram_config import API_ID, API_HASH, PHONE, CHANNELS
except ImportError:
    API_ID, API_HASH, PHONE, CHANNELS = None, None, None, {}


DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "telegram.sqlite"
CSV_FILE = DATA_DIR / "telegram_snapshots.csv"
SESSION_FILE = "telegram_session"
CHECKPOINTS = [5, 10, 20, 30]


def open_db():
    DATA_DIR.mkdir(exist_ok=True)
    db = sqlite3.connect(DB_FILE)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            channel TEXT,
            message_id INTEGER,
            subscribers INTEGER,
            channel_type TEXT,
            published_at TEXT,
            source TEXT DEFAULT 'telegram',
            PRIMARY KEY(channel, message_id)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            channel TEXT,
            message_id INTEGER,
            checkpoint_min INTEGER,
            measured_at TEXT,
            actual_age_min REAL,
            views INTEGER,
            reactions INTEGER,
            forwards INTEGER,
            PRIMARY KEY(channel, message_id, checkpoint_min)
        )
        """
    )
    db.commit()
    return db


def reaction_count(message):
    # reactions گاهی None است، مثلا وقتی کانال واکنش را بسته باشد
    if not message.reactions or not message.reactions.results:
        return 0
    return sum(x.count for x in message.reactions.results)


def save_post(db, channel, message, subscribers, channel_type):
    db.execute(
        "INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?, ?, 'telegram')",
        (channel, message.id, subscribers, channel_type, message.date.isoformat()),
    )


def save_snapshot(db, channel, message, checkpoint, age_min):
    db.execute(
        """
        INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            channel,
            message.id,
            checkpoint,
            datetime.now(timezone.utc).isoformat(),
            round(age_min, 2),
            int(message.views or 0),
            int(reaction_count(message)),
            int(message.forwards or 0),
        ),
    )


def already_saved(db, channel, message_id, checkpoint):
    row = db.execute(
        """
        SELECT 1 FROM snapshots
        WHERE channel=? AND message_id=? AND checkpoint_min=?
        """,
        (channel, message_id, checkpoint),
    ).fetchone()
    return row is not None


async def channel_info(client, username):
    entity = await client.get_entity(username)
    if not getattr(entity, "username", None) or not getattr(entity, "broadcast", False):
        raise ValueError("کانال عمومی broadcast نیست")
    full = await client(functions.channels.GetFullChannelRequest(entity))
    subscribers = int(full.full_chat.participants_count or 0)
    return entity, subscribers


async def prepare_channels(client):
    ready = {}
    for channel, channel_type in CHANNELS.items():
        try:
            entity, subscribers = await channel_info(client, channel)
            ready[channel] = (channel_type, entity, subscribers)
            print(f"@{channel}: {subscribers} عضو")
            await asyncio.sleep(1.2)
        except FloodWaitError as error:
            print(f"تلگرام گفته {error.seconds} ثانیه صبر کنیم...")
            await asyncio.sleep(error.seconds + 2)
        except Exception as error:
            print(f"کانال @{channel} قابل استفاده نبود: {error}")
    return ready


async def scan_early_posts(client, db, channel_cache):
    now = datetime.now(timezone.utc)
    saved_now = 0

    for channel, (channel_type, entity, subscribers) in channel_cache.items():
        try:
            async for message in client.iter_messages(entity, limit=25):
                if message.date is None or message.views is None:
                    continue

                age = (now - message.date).total_seconds() / 60
                if age < 0 or age > 34:
                    continue

                save_post(db, channel, message, subscribers, channel_type)
                for checkpoint in CHECKPOINTS:
                    # سه دقیقه تلورانس گذاشتم که با اینترنت معمولی هم نقطه از دست نرود
                    if checkpoint <= age <= checkpoint + 3:
                        if not already_saved(db, channel, message.id, checkpoint):
                            save_snapshot(db, channel, message, checkpoint, age)
                            saved_now += 1
            db.commit()
            await asyncio.sleep(1.2)
        except FloodWaitError as error:
            print(f"تلگرام گفته {error.seconds} ثانیه صبر کنیم...")
            await asyncio.sleep(error.seconds + 2)
        except Exception as error:
            print(f"مشکل در @{channel}: {error}")

    print(f"این دور {saved_now} اندازه گیری جدید ذخیره شد.")


async def watch(client, db, minutes, final_during_run=False):
    print(f"جمع آوری اولیه برای {minutes} دقیقه شروع شد.")
    # زمان پایان را از روی ساعت واقعی حساب می کنیم، نه تعداد دورها
    loop = asyncio.get_running_loop()
    finish_time = loop.time() + max(1, minutes) * 60
    channel_cache = await prepare_channels(client)
    if not channel_cache:
        print("هیچ کانال معتبری پیدا نشد.")
        return

    last_final_check = loop.time()
    while loop.time() < finish_time:
        await scan_early_posts(client, db, channel_cache)

        # در حالت full، پست های دفعه قبل را هم نزدیک 24 ساعت ثبت می کنیم
        if final_during_run and loop.time() - last_final_check >= 30 * 60:
            await collect_final(client, db)
            last_final_check = loop.time()

        remaining = finish_time - loop.time()
        if remaining > 0:
            await asyncio.sleep(min(60, remaining))

    export_csv(db)


async def collect_final(client, db):
    now = datetime.now(timezone.utc)
    rows = db.execute(
        """
        SELECT p.channel, p.message_id, p.published_at
        FROM posts p
        LEFT JOIN snapshots s
          ON p.channel=s.channel AND p.message_id=s.message_id
         AND s.checkpoint_min=1440
        WHERE s.message_id IS NULL
        """
    ).fetchall()

    count = 0
    entities = {}
    for channel, message_id, published_at in rows:
        age = (now - datetime.fromisoformat(published_at)).total_seconds() / 60
        if age < 1440:  # هنوز 24 ساعت کامل نشده
            continue
        try:
            if channel not in entities:
                entities[channel] = await client.get_entity(channel)
            entity = entities[channel]
            message = await client.get_messages(entity, ids=message_id)
            if message is not None:
                save_snapshot(db, channel, message, 1440, age)
                count += 1
            db.commit()
            await asyncio.sleep(1.2)
        except FloodWaitError as error:
            print(f"تلگرام گفته {error.seconds} ثانیه صبر کنیم...")
            await asyncio.sleep(error.seconds + 2)
        except Exception as error:
            print(f"مشکل در @{channel} پیام {message_id}: {error}")

    export_csv(db)
    print(f"نقطه 24 ساعته برای {count} پست ذخیره شد.")


def export_csv(db):
    query = """
        SELECT p.channel, p.message_id, p.subscribers, p.channel_type,
               p.published_at, s.checkpoint_min, s.measured_at,
               s.actual_age_min, s.views, s.reactions, s.forwards, p.source
        FROM posts p JOIN snapshots s
          ON p.channel=s.channel AND p.message_id=s.message_id
        ORDER BY p.published_at, p.channel, p.message_id, s.checkpoint_min
    """
    rows = db.execute(query).fetchall()
    names = [x[0] for x in db.execute(query).description]
    with CSV_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(names)
        writer.writerows(rows)
    print(f"CSV ساخته شد: {CSV_FILE} ({len(rows)} ردیف)")


def show_status(db):
    counts = db.execute(
        "SELECT checkpoint_min, COUNT(*) FROM snapshots GROUP BY checkpoint_min ORDER BY checkpoint_min"
    ).fetchall()
    early_complete = db.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT channel, message_id FROM snapshots
            WHERE checkpoint_min IN (5, 10, 20, 30)
            GROUP BY channel, message_id
            HAVING COUNT(DISTINCT checkpoint_min)=4
        )
        """
    ).fetchone()[0]
    complete = db.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT channel, message_id FROM snapshots
            GROUP BY channel, message_id HAVING COUNT(DISTINCT checkpoint_min)=5
        )
        """
    ).fetchone()[0]
    channels = db.execute(
        "SELECT channel, MAX(subscribers) FROM posts GROUP BY channel ORDER BY MAX(subscribers)"
    ).fetchall()
    complete_by_channel = db.execute(
        """
        SELECT channel, COUNT(*) FROM (
            SELECT channel, message_id FROM snapshots
            GROUP BY channel, message_id HAVING COUNT(DISTINCT checkpoint_min)=5
        ) GROUP BY channel ORDER BY COUNT(*) DESC
        """
    ).fetchall()
    print("تعداد snapshotها:", dict(counts))
    print("پست با هر چهار نقطه اولیه:", early_complete)
    print("پست کامل با نقطه 24 ساعته:", complete)
    print("پست کامل به تفکیک کانال:", dict(complete_by_channel))
    print("اندازه کانال ها:", channels)
    if channels and not any(n < 10000 for _, n in channels):
        print("هشدار: هنوز کانال زیر 10000 عضو نداریم.")
    if channels and not any(n > 100000 for _, n in channels):
        print("هشدار: هنوز کانال بالای 100000 عضو نداریم.")


async def main(args):
    db = open_db()
    if args.command in ["export", "status"]:
        export_csv(db) if args.command == "export" else show_status(db)
        db.close()
        return

    if not API_ID or not API_HASH or not CHANNELS:
        db.close()
        raise SystemExit(
            "اول telegram_config.example.py را با نام telegram_config.py کپی و کامل کن."
        )

    client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
    await client.start(phone=PHONE)
    try:
        if args.command == "watch":
            await watch(client, db, args.minutes)
        elif args.command == "full":
            await watch(client, db, args.minutes, final_during_run=True)
            await collect_final(client, db)
            show_status(db)
        elif args.command == "final":
            await collect_final(client, db)
    finally:
        await client.disconnect()
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("watch", help="ثبت نقاط 5 تا 30 دقیقه")
    first.add_argument("--minutes", type=int, default=40)
    full = sub.add_parser("full", help="جمع آوری اولیه و سپس ثبت خودکار نقطه 24 ساعته")
    full.add_argument("--minutes", type=int, default=1620)
    sub.add_parser("final", help="ثبت نقطه 24 ساعت در اجرای روز بعد")
    sub.add_parser("export", help="فقط خروجی CSV از SQLite")
    sub.add_parser("status", help="دیدن تعداد داده های جمع شده")
    asyncio.run(main(parser.parse_args()))
