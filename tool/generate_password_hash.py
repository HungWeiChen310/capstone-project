#!/usr/bin/env python3
import sys
from werkzeug.security import generate_password_hash

def main():
    if len(sys.argv) != 2:
        print("Usage: python tool/generate_password_hash.py <password>")
        sys.exit(1)

    password = sys.argv[1]
    hashed_password = generate_password_hash(password)
    print(f"Password: {password}")
    print(f"Hash: {hashed_password}")
    print("\nPlease set the following in your .env file:")
    print(f"ADMIN_PASSWORD_HASH={hashed_password}")

if __name__ == "__main__":
    main()
