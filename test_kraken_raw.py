import http.client
import urllib.request
import urllib.parse
import hashlib
import hmac
import base64
import json
import time

def main():
    print("--- RAW Kraken Futures Test ---")
    
    # Load Keys
    try:
        with open('dist/user_config.json', 'r') as f:
            config = json.load(f)
            public_key = config.get('api_key', '').strip()
            private_key = config.get('secret', '').strip()
    except Exception as e:
        print(f"Error loading config: {e}")
        # Try root if dist fails
        try:
             with open('user_config.json', 'r') as f:
                config = json.load(f)
                public_key = config.get('api_key', '').strip()
                private_key = config.get('secret', '').strip()
        except:
            print("Could not load keys.")
            return

    if not public_key or not private_key:
        print("Keys are empty.")
        return

    print(f"Testing Key: {public_key[:6]}...")

    # Test /accounts endpoint (Safe, GET)
    # Docs: https://docs.kraken.com/api/docs/futures-api/trading/get-accounts
    # Path: /derivatives/api/v3/accounts
    
    try:
        response = request(
            method="GET",
            path="/derivatives/api/v3/accounts",
            query={}, 
            public_key=public_key,
            private_key=private_key,
            environment="https://demo-futures.kraken.com"
        )
        print("\nServer Response:")
        print(response.read().decode())
        print("\n✅ If you see a JSON with 'accounts' or 'result': success, keys are working!")
        
    except Exception as e:
        print(f"\n❌ Request Failed: {e}")
        try:
             if hasattr(e, 'read'):
                 print(e.read().decode())
        except:
            pass

# --- User Provided Logic Below ---

def request(method: str = "GET", path: str = "", query: dict | None = None, body: dict | None = None, nonce: str = "", public_key: str = "", private_key: str = "", environment: str = "") -> http.client.HTTPResponse:
    if not nonce:
        nonce = str(int(time.time() * 1000)) # Generate nonce if empty

    url = environment + path
    query_str = ""
    if query is not None and len(query) > 0:
        query_str = urllib.parse.urlencode(query)
        url += "?" + query_str
    
    body_str = ""
    if body is not None and len(body) > 0:
        body_str = urllib.parse.urlencode(body)
    
    headers = {}
    if len(public_key) > 0:
        headers["APIKey"] = public_key
        # User logic for signature
        # Note: headers["Nonce"] IS sent in user snippet
        headers["Nonce"] = nonce
        headers["Authent"] = get_signature(private_key, query_str+body_str, nonce, path)

    # For GET, usually bodies are empty.
    data_bytes = body_str.encode() if body_str else None

    req = urllib.request.Request(
        method=method,
        url=url,
        data=data_bytes,
        headers=headers,
    )
    return urllib.request.urlopen(req)

def get_signature(private_key: str, data: str, nonce: str, path: str) -> str:
    # User provided logic:
    # (data + nonce + path.removeprefix("/derivatives")).encode()
    
    # handle removeprefix for older python versions if needed (3.14 has it)
    endpoint = path
    if endpoint.startswith("/derivatives"):
        endpoint = endpoint[len("/derivatives"):]
        
    message = (data + nonce + endpoint).encode()
    
    return sign(
        private_key=private_key,
        message=hashlib.sha256(message).digest()
    )

def sign(private_key: str, message: bytes) -> str:
    return base64.b64encode(
        hmac.new(
            key=base64.b64decode(private_key),
            msg=message,
            digestmod=hashlib.sha512,
        ).digest()
    ).decode()

if __name__ == "__main__":
    main()
