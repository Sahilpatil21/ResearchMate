"""CLI script to migrate local research papers to Cloudinary storage.

Usage:
    python scripts/migrate_to_cloudinary.py [--user USER_ID] [--delete-local]

Example:
    python scripts/migrate_to_cloudinary.py
    python scripts/migrate_to_cloudinary.py --user usr_1234567890 --delete-local
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage.cloudinary_storage import get_cloudinary_storage
from src.storage.migration import CloudinaryMigrationManager
from src.storage.mongo_storage import get_mongo_storage


def main():
    parser = argparse.ArgumentParser(description="Migrate local ResearchMate PDF papers to Cloudinary.")
    parser.add_argument("--user", type=str, default=None, help="Specific user_id to migrate (default: all users)")
    parser.add_argument("--delete-local", action="store_true", help="Delete local PDF copies after verified upload")
    args = parser.parse_args()

    c_storage = get_cloudinary_storage()
    if not c_storage.is_configured:
        print("❌ Error: Cloudinary is not configured in .env.")
        print("Please set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET.")
        sys.exit(1)

    m_storage = get_mongo_storage()
    manager = CloudinaryMigrationManager(cloudinary_storage=c_storage, mongo_storage=m_storage)

    print("🚀 Starting Cloudinary Migration...")
    print(f"• Cloudinary Cloud Name: {c_storage.cloud_name}")
    print(f"• Delete Local PDFs After Upload: {args.delete_local}")

    if args.user:
        print(f"\nMigrating papers for user: {args.user}")
        result = manager.migrate_user_papers(user_id=args.user, delete_local=args.delete_local)
        print(f"✅ Found: {result['total_found']} | Migrated: {result['migrated']} | Failed: {result['failed']}")
        for item in result["details"]:
            print(f"  - {item['filename']} -> {item['status']} (ID: {item.get('cloudinary_public_id')})")
    else:
        print("\nScanning all users in data/users/...")
        results = manager.migrate_all_users(delete_local=args.delete_local)
        total_migrated = sum(r["migrated"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        print(f"\n🎉 Migration Complete: {total_migrated} papers migrated, {total_failed} failed across {len(results)} users.")


if __name__ == "__main__":
    main()
