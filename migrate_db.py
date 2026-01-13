
from sqlalchemy import create_engine, text
import config

def migrate():
    engine = create_engine(config.DB_URL)
    with engine.connect() as conn:
        try:
            # Check if column exists (simple try/catch approach or PRAGMA)
            # SQLite specific
            conn.execute(text("ALTER TABLE strategies ADD COLUMN status VARCHAR DEFAULT 'lab'"))
            print("✅ Migration Successful: Added 'status' column to strategies.")
        except Exception as e:
            if "duplicate column name" in str(e):
                print("ℹ️ Column 'status' already exists.")
            else:
                print(f"❌ Migration Failed: {e}")

if __name__ == "__main__":
    migrate()
