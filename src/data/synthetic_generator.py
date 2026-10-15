"""Deterministic synthetic e-commerce catalog generator."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from src.config import Settings, get_settings
from src.schemas.product import Product

CATEGORIES: dict[str, dict] = {
    "headphones": {
        "attrs": ["connection", "battery_hours", "color", "weight_g", "noise_cancellation"],
        "tags": ["wireless", "travel", "noise-cancelling", "gaming", "commute"],
        "price_range": (29.0, 399.0),
    },
    "laptops": {
        "attrs": ["ram_gb", "storage_gb", "screen_inches", "weight_g", "battery_hours"],
        "tags": ["portable", "work", "student", "creator"],
        "price_range": (399.0, 2499.0),
    },
    "smartphones": {
        "attrs": ["storage_gb", "battery_hours", "color", "weight_g", "waterproof"],
        "tags": ["5g", "camera", "compact", "flagship"],
        "price_range": (149.0, 1599.0),
    },
    "tablets": {
        "attrs": ["storage_gb", "screen_inches", "battery_hours", "color", "weight_g"],
        "tags": ["drawing", "media", "portable"],
        "price_range": (129.0, 1299.0),
    },
    "keyboards": {
        "attrs": ["connection", "layout", "color", "weight_g", "mechanical"],
        "tags": ["office", "gaming", "compact", "quiet"],
        "price_range": (19.0, 249.0),
    },
    "mice": {
        "attrs": ["connection", "dpi", "color", "weight_g", "ergonomic"],
        "tags": ["gaming", "office", "travel"],
        "price_range": (9.0, 149.0),
    },
    "monitors": {
        "attrs": ["screen_inches", "refresh_hz", "resolution", "color", "weight_g"],
        "tags": ["ultrawide", "office", "gaming", "design"],
        "price_range": (99.0, 999.0),
    },
    "speakers": {
        "attrs": ["connection", "battery_hours", "color", "waterproof", "weight_g"],
        "tags": ["portable", "home", "party", "outdoor"],
        "price_range": (19.0, 499.0),
    },
    "chargers": {
        "attrs": ["watts", "ports", "color", "foldable", "weight_g"],
        "tags": ["travel", "fast-charge", "usb-c"],
        "price_range": (9.0, 89.0),
    },
    "cases": {
        "attrs": ["compatible_model", "color", "material", "drop_protection", "weight_g"],
        "tags": ["protection", "slim", "rugged"],
        "price_range": (9.0, 79.0),
    },
    "cameras": {
        "attrs": ["sensor_mp", "video_4k", "color", "weight_g", "battery_hours"],
        "tags": ["vlog", "travel", "mirrorless"],
        "price_range": (199.0, 2999.0),
    },
    "smartwatches": {
        "attrs": ["battery_hours", "color", "waterproof", "gps", "weight_g"],
        "tags": ["fitness", "sleep", "notification"],
        "price_range": (49.0, 699.0),
    },
    "routers": {
        "attrs": ["wifi_standard", "ports", "color", "mesh", "weight_g"],
        "tags": ["home", "gaming", "mesh"],
        "price_range": (39.0, 399.0),
    },
    "webcams": {
        "attrs": ["resolution", "microphone", "color", "autofocus", "weight_g"],
        "tags": ["remote-work", "streaming"],
        "price_range": (29.0, 249.0),
    },
    "earbuds": {
        "attrs": ["connection", "battery_hours", "color", "noise_cancellation", "weight_g"],
        "tags": ["wireless", "sport", "commute"],
        "price_range": (19.0, 349.0),
    },
}

BRANDS = [
    "Northstar Audio",
    "Alpine Circuit",
    "Blue Harbor",
    "Cedar Logic",
    "Delta Forge",
    " Ember Peak",
    "Frostbyte",
    "Granite Labs",
    "Helio Devices",
    "Ion Grove",
    "Jade Signal",
    "Kite Systems",
    "Lumen Oak",
    "Maple Compute",
    "Nova Ridge",
    "Orbit Nest",
    "Pine Pixel",
    "Quartz Soft",
    "River Chip",
    "Solstice Gear",
]
# fix accidental space in brand list
BRANDS = [b.strip() for b in BRANDS]

COLORS = ["black", "white", "silver", "blue", "red", "green", "graphite"]
CONNECTIONS = ["Bluetooth", "USB-C", "USB-A", "2.4GHz", "wired"]
DEVICE_FAMILIES = [f"device_family_{i}" for i in range(1, 16)]


def _sku(category: str, model: str, color: str, rng: random.Random) -> str:
    prefix = category[:5].upper()
    return f"{prefix}-{model}-{color[:3].upper()}{rng.randint(0, 9)}"


def _attrs_for(category: str, rng: random.Random, incomplete: bool = False) -> dict:
    spec = CATEGORIES[category]
    attrs: dict = {}
    for name in spec["attrs"]:
        if incomplete and rng.random() < 0.35:
            continue
        if name == "connection":
            attrs[name] = rng.choice(CONNECTIONS)
        elif name == "battery_hours":
            attrs[name] = rng.choice([8, 12, 20, 24, 30, 36, 38, 40, 48])
        elif name == "color":
            attrs[name] = rng.choice(COLORS)
        elif name == "weight_g":
            attrs[name] = rng.randint(15, 2200)
        elif name in {"noise_cancellation", "waterproof", "mechanical", "ergonomic", "video_4k", "gps", "mesh", "microphone", "autofocus", "foldable", "drop_protection"}:
            attrs[name] = rng.choice([True, False])
        elif name == "ram_gb":
            attrs[name] = rng.choice([8, 16, 32, 64])
        elif name == "storage_gb":
            attrs[name] = rng.choice([64, 128, 256, 512, 1024])
        elif name == "screen_inches":
            attrs[name] = rng.choice([10.1, 13.3, 14.0, 15.6, 27.0, 32.0])
        elif name == "dpi":
            attrs[name] = rng.choice([800, 1600, 3200, 6400])
        elif name == "refresh_hz":
            attrs[name] = rng.choice([60, 75, 144, 165, 240])
        elif name == "resolution":
            attrs[name] = rng.choice(["1080p", "1440p", "4K"])
        elif name == "watts":
            attrs[name] = rng.choice([20, 30, 45, 65, 100])
        elif name == "ports":
            attrs[name] = rng.randint(1, 4)
        elif name == "layout":
            attrs[name] = rng.choice(["ANSI", "ISO", "60%", "TKL"])
        elif name == "material":
            attrs[name] = rng.choice(["silicone", "plastic", "leather", "tpu"])
        elif name == "compatible_model":
            attrs[name] = rng.choice(DEVICE_FAMILIES)
        elif name == "sensor_mp":
            attrs[name] = rng.choice([12, 20, 24, 33, 45])
        elif name == "wifi_standard":
            attrs[name] = rng.choice(["Wi-Fi 5", "Wi-Fi 6", "Wi-Fi 6E"])
        else:
            attrs[name] = rng.choice(["standard", "pro", "lite"])
    return attrs


def generate_catalog(settings: Settings | None = None) -> list[Product]:
    settings = settings or get_settings()
    rng = random.Random(settings.dataset_seed)
    categories = list(CATEGORIES.keys())
    brands = BRANDS[:]
    # pad brands to ~50-80 unique-ish names
    while len(brands) < 60:
        brands.append(f"Brand-{len(brands):03d}")

    products: list[Product] = []
    now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)

    for i in range(settings.catalog_size):
        category = categories[i % len(categories)]
        brand = brands[i % len(brands)]
        model = f"X{100 + (i % 200)}"
        color = COLORS[i % len(COLORS)]
        incomplete = i % 17 == 0
        attrs = _attrs_for(category, rng, incomplete=incomplete)
        if "color" not in attrs:
            attrs["color"] = color

        lo, hi = CATEGORIES[category]["price_range"]
        price = round(rng.uniform(lo, hi), 2)
        # price edge cases near round thresholds
        if i % 23 == 0:
            price = 99.99
        elif i % 29 == 0:
            price = 100.00
        elif i % 31 == 0:
            price = 149.99

        in_stock = i % 11 != 0
        sku = _sku(category, model, attrs.get("color", color), rng)
        # near-duplicate SKU / title cases
        if i > 0 and i % 41 == 0:
            prev = products[i - 1]
            sku = prev.sku[:-1] + ("A" if prev.sku[-1] != "A" else "B")
            title = prev.title + " Alt"
        else:
            title = f"{model} {category.replace('_', ' ').title()} {attrs.get('color', color).title()}"

        tags = rng.sample(CATEGORIES[category]["tags"], k=min(3, len(CATEGORIES[category]["tags"])))
        compat = [rng.choice(DEVICE_FAMILIES)] if category in {"cases", "chargers", "earbuds", "headphones"} else []
        related: list[str] = []
        if i > 2 and i % 7 == 0:
            related = [products[i - 2].product_id]

        desc_en = (
            f"{brand} {title} for everyday use. "
            f"Designed for {tags[0]} scenarios with reliable build quality."
        )
        if i % 13 == 0:
            desc_en += " Version française disponible: appareil compact et léger."

        product = Product(
            product_id=f"prod_{i:06d}",
            sku=sku,
            title=title,
            brand=brand,
            category=category,
            description=desc_en,
            attributes=attrs,
            price=price,
            currency="EUR",
            in_stock=in_stock,
            rating=round(rng.uniform(2.5, 5.0), 1),
            tags=tags,
            compatible_with=compat,
            related_product_ids=related,
            source_record=f"synthetic_catalog_{settings.catalog_version}",
            updated_at=now,
        )
        products.append(product)

    # Ensure a few handcrafted hard cases exist at the front for demos
    if products:
        products[0] = products[0].model_copy(
            update={
                "product_id": "prod_000000",
                "sku": "AUDIO-X200-BLK",
                "title": "X200 Wireless Noise-Cancelling Headphones",
                "brand": "Northstar Audio",
                "category": "headphones",
                "description": (
                    "Over-ear wireless headphones with active noise cancellation "
                    "for long flights and commute."
                ),
                "attributes": {
                    "connection": "Bluetooth",
                    "battery_hours": 38,
                    "color": "black",
                    "weight_g": 245,
                    "noise_cancellation": True,
                },
                "price": 129.99,
                "in_stock": True,
                "tags": ["wireless", "travel", "noise-cancelling"],
                "compatible_with": ["device_family_7"],
            }
        )
        products[1] = products[1].model_copy(
            update={
                "product_id": "prod_000001",
                "sku": "AUDIO-X210-BLK",
                "title": "X210 Wireless Noise-Cancelling Headphones",
                "brand": "Northstar Audio",
                "category": "headphones",
                "description": "Similar to X200 with longer battery and slightly higher weight.",
                "attributes": {
                    "connection": "Bluetooth",
                    "battery_hours": 42,
                    "color": "black",
                    "weight_g": 260,
                    "noise_cancellation": True,
                },
                "price": 159.99,
                "in_stock": True,
                "tags": ["wireless", "travel", "noise-cancelling"],
                "compatible_with": ["device_family_7"],
                "related_product_ids": ["prod_000000"],
            }
        )
        # near-duplicate SKU trap
        products[2] = products[2].model_copy(
            update={
                "product_id": "prod_000002",
                "sku": "AUDIO-X200-BLJ",
                "title": "X200 Wireless Headphones Lite",
                "brand": "Northstar Audio",
                "category": "headphones",
                "price": 99.99,
                "in_stock": False,
                "attributes": {
                    "connection": "Bluetooth",
                    "battery_hours": 20,
                    "color": "black",
                    "weight_g": 220,
                    "noise_cancellation": False,
                },
            }
        )
        # accessory for device family 7
        products[3] = products[3].model_copy(
            update={
                "product_id": "prod_000003",
                "sku": "CASE-DF7-BLK",
                "title": "Rugged Case for Device Family 7",
                "brand": "Solstice Gear",
                "category": "cases",
                "price": 39.99,
                "in_stock": True,
                "compatible_with": ["device_family_7"],
                "attributes": {
                    "compatible_model": "device_family_7",
                    "color": "black",
                    "material": "tpu",
                    "drop_protection": True,
                    "weight_g": 45,
                },
            }
        )

    return products


def write_catalog(products: list[Product], settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
    out = settings.catalog_path
    with out.open("w", encoding="utf-8") as fh:
        for product in products:
            fh.write(product.model_dump_json() + "\n")

    meta = {
        "catalog_version": settings.catalog_version,
        "seed": settings.dataset_seed,
        "count": len(products),
        "source_record": f"synthetic_catalog_{settings.catalog_version}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    settings.catalog_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic catalog JSONL")
    parser.add_argument("--size", type=int, default=None, help="Override catalog size")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.size is not None:
        settings.catalog_size = args.size
    if args.seed is not None:
        settings.dataset_seed = args.seed

    products = generate_catalog(settings)
    path = write_catalog(products, settings)
    print(f"Wrote {len(products)} products to {path}")


if __name__ == "__main__":
    main()
