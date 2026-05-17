"""Phase 2 seeder — 10 000 items + 1 000 users + MiniLM embeddings + ivfflat index.

Idempotent: early-returns when `item` already has >= TARGET_ITEMS rows. For a
clean re-seed: `make down-v && make up && make migrate && make seed`.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
from faker import Faker
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.config import get_settings
from click_rec.db.base import dispose_engine, get_sessionmaker
from click_rec.models import USER_SEGMENTS, Item, UserAccount

logger = logging.getLogger(__name__)

TARGET_ITEMS = 10_000
TARGET_USERS = 1_000
ITEMS_PER_CATEGORY = 200
USERS_PER_SEGMENT = 200
BULK_CHUNK = 500
EMBED_BATCH = 256
RANDOM_SEED = 42
COUNTRIES = ("US", "GB", "DE", "FR", "JP", "CA", "AU", "BR", "IN", "SE")

ADJECTIVES = (
    "Premium",
    "Compact",
    "Wireless",
    "Portable",
    "Professional",
    "Classic",
    "Modern",
    "Sleek",
    "Rugged",
    "Ergonomic",
    "Lightweight",
    "Durable",
    "Elite",
    "Advanced",
    "Smart",
    "Deluxe",
    "Eco",
    "Vintage",
    "Essential",
    "Ultimate",
)

# (category_path, product_noun, brands, (price_low, price_high))
CategoryRow = tuple[str, str, tuple[str, ...], tuple[float, float]]

CATEGORIES: tuple[CategoryRow, ...] = (
    (
        "electronics/headphones",
        "Headphones",
        ("Sony", "Bose", "Sennheiser", "JBL", "Anker"),
        (20.0, 400.0),
    ),
    (
        "electronics/earbuds",
        "Earbuds",
        ("Apple", "Samsung", "Jabra", "Beats", "Soundcore"),
        (15.0, 300.0),
    ),
    ("electronics/speakers", "Speaker", ("JBL", "Bose", "Sonos", "Marshall", "UE"), (25.0, 800.0)),
    ("electronics/laptops", "Laptop", ("Apple", "Dell", "HP", "Lenovo", "ASUS"), (400.0, 3000.0)),
    (
        "electronics/keyboards",
        "Keyboard",
        ("Logitech", "Keychron", "Razer", "Corsair", "HyperX"),
        (30.0, 300.0),
    ),
    (
        "electronics/mice",
        "Mouse",
        ("Logitech", "Razer", "Apple", "SteelSeries", "Glorious"),
        (15.0, 180.0),
    ),
    ("electronics/monitors", "Monitor", ("LG", "Dell", "Samsung", "ASUS", "BenQ"), (120.0, 1500.0)),
    (
        "electronics/smartphones",
        "Phone",
        ("Apple", "Samsung", "Google", "OnePlus", "Motorola"),
        (200.0, 1600.0),
    ),
    (
        "electronics/tablets",
        "Tablet",
        ("Apple", "Samsung", "Lenovo", "Microsoft", "Amazon"),
        (100.0, 1200.0),
    ),
    (
        "electronics/smartwatches",
        "Smartwatch",
        ("Apple", "Garmin", "Fitbit", "Samsung", "Amazfit"),
        (50.0, 900.0),
    ),
    (
        "home/coffee-makers",
        "Coffee Maker",
        ("Breville", "DeLonghi", "Keurig", "Ninja", "Nespresso"),
        (40.0, 800.0),
    ),
    (
        "home/blenders",
        "Blender",
        ("Vitamix", "Ninja", "Blendtec", "KitchenAid", "Oster"),
        (25.0, 700.0),
    ),
    (
        "home/toasters",
        "Toaster",
        ("Breville", "Cuisinart", "KitchenAid", "Hamilton Beach", "Dash"),
        (20.0, 300.0),
    ),
    (
        "home/air-fryers",
        "Air Fryer",
        ("Ninja", "Instant", "Cosori", "Philips", "Chefman"),
        (50.0, 400.0),
    ),
    (
        "home/stand-mixers",
        "Stand Mixer",
        ("KitchenAid", "Cuisinart", "Breville", "Bosch", "Hamilton Beach"),
        (150.0, 900.0),
    ),
    (
        "home/kettles",
        "Kettle",
        ("Breville", "Cuisinart", "Hamilton Beach", "OXO", "Fellow"),
        (20.0, 250.0),
    ),
    (
        "home/cookware",
        "Cookware Set",
        ("All-Clad", "Le Creuset", "Lodge", "Calphalon", "T-fal"),
        (50.0, 1200.0),
    ),
    (
        "home/cutlery",
        "Knife Set",
        ("Wusthof", "Shun", "Victorinox", "Zwilling", "Global"),
        (30.0, 800.0),
    ),
    (
        "home/dinnerware",
        "Dinnerware Set",
        ("Corelle", "Mikasa", "Lenox", "Gibson", "Fiesta"),
        (25.0, 500.0),
    ),
    (
        "home/food-storage",
        "Food Storage Set",
        ("OXO", "Rubbermaid", "Pyrex", "Snapware", "Glasslock"),
        (15.0, 200.0),
    ),
    ("home/lamps", "Lamp", ("Philips", "IKEA", "Govee", "Brightech", "Adesso"), (20.0, 350.0)),
    ("home/rugs", "Rug", ("Safavieh", "Ruggable", "Loloi", "Nuloom", "Jaipur"), (40.0, 900.0)),
    (
        "home/throw-pillows",
        "Throw Pillow",
        ("Pottery Barn", "West Elm", "Target", "Hearth & Hand", "Amazon Basics"),
        (15.0, 150.0),
    ),
    (
        "home/wall-art",
        "Wall Art",
        ("Society6", "Minted", "Etsy", "UGallery", "IKEA"),
        (20.0, 500.0),
    ),
    (
        "home/curtains",
        "Curtain Set",
        ("IKEA", "Pottery Barn", "Deconovo", "Amazon Basics", "Nicetown"),
        (20.0, 300.0),
    ),
    (
        "furniture/office-chairs",
        "Office Chair",
        ("Herman Miller", "Steelcase", "Secretlab", "IKEA", "Branch"),
        (150.0, 2000.0),
    ),
    ("furniture/desks", "Desk", ("IKEA", "Uplift", "Fully", "Flexispot", "Vari"), (120.0, 1500.0)),
    (
        "furniture/bookshelves",
        "Bookshelf",
        ("IKEA", "Sauder", "Prepac", "CB2", "West Elm"),
        (60.0, 900.0),
    ),
    (
        "furniture/nightstands",
        "Nightstand",
        ("IKEA", "CB2", "West Elm", "Sauder", "Walker Edison"),
        (40.0, 700.0),
    ),
    ("furniture/sofas", "Sofa", ("IKEA", "Article", "West Elm", "CB2", "Ashley"), (300.0, 3500.0)),
    (
        "apparel/mens-running-shoes",
        "Running Shoes",
        ("Nike", "Adidas", "Brooks", "Hoka", "Asics"),
        (60.0, 250.0),
    ),
    (
        "apparel/mens-sneakers",
        "Sneakers",
        ("Nike", "Adidas", "New Balance", "Puma", "Vans"),
        (50.0, 220.0),
    ),
    (
        "apparel/mens-t-shirts",
        "T-Shirt",
        ("Uniqlo", "Nike", "Adidas", "Everlane", "Patagonia"),
        (12.0, 80.0),
    ),
    (
        "apparel/mens-jackets",
        "Jacket",
        ("Patagonia", "North Face", "Arc'teryx", "Columbia", "Carhartt"),
        (60.0, 700.0),
    ),
    ("apparel/mens-jeans", "Jeans", ("Levi's", "Wrangler", "Lee", "Gap", "Mavi"), (35.0, 250.0)),
    (
        "apparel/womens-dresses",
        "Dress",
        ("Zara", "H&M", "Everlane", "Reformation", "Aritzia"),
        (30.0, 400.0),
    ),
    (
        "apparel/womens-handbags",
        "Handbag",
        ("Coach", "Kate Spade", "Michael Kors", "Longchamp", "Fossil"),
        (40.0, 1200.0),
    ),
    (
        "apparel/womens-boots",
        "Boots",
        ("Dr. Martens", "Timberland", "Sorel", "UGG", "Blundstone"),
        (70.0, 400.0),
    ),
    (
        "apparel/womens-sweaters",
        "Sweater",
        ("Uniqlo", "Everlane", "J.Crew", "Madewell", "Gap"),
        (30.0, 300.0),
    ),
    (
        "apparel/womens-activewear",
        "Leggings",
        ("Lululemon", "Nike", "Alo", "Athleta", "Fabletics"),
        (25.0, 250.0),
    ),
    (
        "beauty/skincare",
        "Skincare Set",
        ("CeraVe", "The Ordinary", "La Roche-Posay", "Paula's Choice", "Kiehl's"),
        (15.0, 300.0),
    ),
    (
        "beauty/haircare",
        "Hair Care Set",
        ("Olaplex", "Living Proof", "Redken", "Moroccanoil", "Pureology"),
        (20.0, 250.0),
    ),
    (
        "beauty/makeup",
        "Makeup Palette",
        ("Urban Decay", "MAC", "Fenty", "NARS", "Maybelline"),
        (15.0, 200.0),
    ),
    (
        "beauty/fragrance",
        "Fragrance",
        ("Chanel", "Dior", "Tom Ford", "Jo Malone", "Le Labo"),
        (40.0, 600.0),
    ),
    (
        "beauty/shaving",
        "Shaving Kit",
        ("Harry's", "Gillette", "Schick", "Dollar Shave Club", "Philips"),
        (15.0, 250.0),
    ),
    (
        "sports/yoga-mats",
        "Yoga Mat",
        ("Manduka", "Lululemon", "Gaiam", "Liforme", "Jade"),
        (20.0, 150.0),
    ),
    (
        "sports/dumbbells",
        "Dumbbell Set",
        ("Bowflex", "PowerBlock", "Rogue", "CAP Barbell", "Yes4All"),
        (30.0, 900.0),
    ),
    (
        "sports/resistance-bands",
        "Resistance Bands",
        ("TheraBand", "Fit Simplify", "Bodylastics", "WODFitters", "Tribe"),
        (10.0, 90.0),
    ),
    (
        "sports/treadmills",
        "Treadmill",
        ("NordicTrack", "Sole", "Peloton", "ProForm", "Horizon"),
        (400.0, 3500.0),
    ),
    (
        "sports/bicycles",
        "Bicycle",
        ("Trek", "Specialized", "Giant", "Cannondale", "Schwinn"),
        (200.0, 4000.0),
    ),
)


def _log_uniform_price(rng: np.random.Generator, low: float, high: float) -> Decimal:
    log_price = rng.uniform(np.log(low), np.log(high))
    return Decimal(f"{np.exp(log_price):.2f}")


def _generate_item_rows(
    rng: random.Random,
    np_rng: np.random.Generator,
    fake: Faker,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    now = datetime.now(tz=UTC)
    window = timedelta(days=730)  # 2 years

    for path, noun, brands, (price_low, price_high) in CATEGORIES:
        for i in range(ITEMS_PER_CATEGORY):
            brand = rng.choice(brands)
            adjective = rng.choice(ADJECTIVES)
            model_code = f"{fake.lexify('?').upper()}{rng.randint(1, 999)}"
            # Pick one of three title templates for variety.
            template = rng.choice((0, 1, 2))
            if template == 0:
                title = f"{adjective} {brand} {noun}"
            elif template == 1:
                title = f"{brand} {model_code} {noun}"
            else:
                title = f"{adjective} {brand} {model_code} {noun}"

            colour = fake.safe_color_name()
            descriptor = " ".join(fake.words(nb=6, unique=True))
            description = (
                f"{title} in {colour}. {descriptor.capitalize()}. "
                f"Built by {brand} for {path.split('/')[-1].replace('-', ' ')} enthusiasts."
            )

            created_at = now - timedelta(days=rng.random() * window.days)
            item_id = f"i_{path.replace('/', '_')}_{i:04d}"

            rows.append(
                {
                    "id": item_id,
                    "title": title,
                    "description": description,
                    "category": path,
                    "brand": brand,
                    "price": _log_uniform_price(np_rng, price_low, price_high),
                    "created_at": created_at,
                    "popularity_score": 0.0,
                }
            )
    return rows


def _generate_user_rows(rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    now = datetime.now(tz=UTC)
    max_age = timedelta(days=365 * 3)

    for segment in USER_SEGMENTS:
        for i in range(USERS_PER_SEGMENT):
            created_at = now - timedelta(days=rng.random() * max_age.days)
            last_active_at = created_at + timedelta(
                days=rng.random() * max(1.0, (now - created_at).days)
            )
            rows.append(
                {
                    "id": f"u_{segment}_{i:04d}",
                    "segment": segment,
                    "country": rng.choice(COUNTRIES),
                    "created_at": created_at,
                    "last_active_at": last_active_at,
                }
            )
    return rows


def _embed_items(rows: list[dict[str, Any]]) -> None:
    settings = get_settings()
    logger.info("loading sentence-transformers model: %s", settings.embedding_model)
    # Imported lazily so `click_rec` tests / CLI `--help` don't pay the load cost.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)
    texts = [f"{row['title']}. {row['description']}" for row in rows]

    logger.info("encoding %d items (batch_size=%d)...", len(texts), EMBED_BATCH)
    vectors = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    for row, vec in zip(rows, vectors, strict=True):
        row["embedding"] = vec.tolist()


async def _bulk_insert_items(session: AsyncSession, rows: list[dict[str, Any]]) -> None:
    for start in range(0, len(rows), BULK_CHUNK):
        chunk = rows[start : start + BULK_CHUNK]
        stmt = pg_insert(Item).on_conflict_do_nothing(index_elements=["id"])
        await session.execute(stmt, chunk)
        logger.info("inserted items %d/%d", min(start + BULK_CHUNK, len(rows)), len(rows))
    await session.commit()


async def _bulk_insert_users(session: AsyncSession, rows: list[dict[str, Any]]) -> None:
    for start in range(0, len(rows), BULK_CHUNK):
        chunk = rows[start : start + BULK_CHUNK]
        stmt = pg_insert(UserAccount).on_conflict_do_nothing(index_elements=["id"])
        await session.execute(stmt, chunk)
    await session.commit()
    logger.info("inserted %d users", len(rows))


async def _write_seeded_user_ids(session: AsyncSession) -> None:
    """Phase 8c: dump existing user_ids to `artifacts/seeded_users.txt` for Locust.

    Reads from the DB so this works on both first-time and re-seed runs:
    a developer who deletes the file then re-runs `make seed` against an
    already-populated catalogue still gets a fresh fixture. Best-effort —
    a write failure only means the loadtest must be pointed at a custom
    file via the SEEDED_USERS env.
    """
    from pathlib import Path

    out = Path(get_settings().eval_output_dir) / "seeded_users.txt"
    res = await session.execute(select(UserAccount.id).order_by(UserAccount.id))
    user_ids = list(res.scalars().all())
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for uid in user_ids:
                fh.write(f"{uid}\n")
        logger.info("wrote %d user ids to %s", len(user_ids), out)
    except OSError as exc:  # noqa: BLE001
        logger.warning("could not write seeded_users.txt: %s", exc)


async def _rebuild_ivfflat_index(session: AsyncSession) -> None:
    # Drop-and-recreate so centroids are trained on the populated table — an
    # index built on empty data has garbage centroids and ruins recall.
    await session.execute(text("DROP INDEX IF EXISTS item_embedding_ivfflat"))
    await session.execute(
        text(
            "CREATE INDEX item_embedding_ivfflat "
            "ON item USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )
    )
    await session.commit()
    logger.info("ivfflat index rebuilt on populated item table")


async def run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sessionmaker = get_sessionmaker()

    async with sessionmaker() as session:
        current = (await session.execute(select(func.count()).select_from(Item))).scalar_one()

    if current >= TARGET_ITEMS:
        logger.info("catalogue already seeded (%d items) — skipping", current)
        async with sessionmaker() as session:
            index_present = (
                await session.execute(
                    text("SELECT 1 FROM pg_class WHERE relname = 'item_embedding_ivfflat'")
                )
            ).scalar_one_or_none()
            if index_present:
                logger.info("ivfflat index already present — skipping rebuild")
            else:
                await _rebuild_ivfflat_index(session)
    else:
        rng = random.Random(RANDOM_SEED)
        np_rng = np.random.default_rng(RANDOM_SEED)
        fake = Faker()
        Faker.seed(RANDOM_SEED)

        logger.info("generating %d items across %d categories...", TARGET_ITEMS, len(CATEGORIES))
        item_rows = _generate_item_rows(rng, np_rng, fake)
        _embed_items(item_rows)

        async with sessionmaker() as session:
            await _bulk_insert_items(session, item_rows)

        logger.info("generating %d users across %d segments...", TARGET_USERS, len(USER_SEGMENTS))
        user_rows = _generate_user_rows(rng)
        async with sessionmaker() as session:
            await _bulk_insert_users(session, user_rows)

        async with sessionmaker() as session:
            await _rebuild_ivfflat_index(session)

        logger.info("seed complete: %d items, %d users", len(item_rows), len(user_rows))

    # Always (re)write the locust user-id fixture from DB so a developer
    # who deletes artifacts/seeded_users.txt and re-runs `make seed`
    # against an already-populated catalogue still gets a fresh file.
    async with sessionmaker() as session:
        await _write_seeded_user_ids(session)

    await dispose_engine()
    return 0
