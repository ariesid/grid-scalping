# gate_grid_bot.py
# Version: 2.1
# Gate.io Grid Trading Bot - Enhanced with Auto-Recovery

import os, time, json, hmac, hashlib, requests, threading
from pathlib import Path
from dotenv import load_dotenv
from decimal import Decimal, ROUND_DOWN

env_path = Path(__file__).with_name("config.env")  # ← baca config.env
load_dotenv(dotenv_path=env_path, override=True)

API_KEY    = os.getenv("GATE_API_KEY")
API_SECRET = os.getenv("GATE_API_SECRET")
USE_TESTNET = os.getenv("USE_TESTNET", "1") == "1"

if not API_KEY or not API_SECRET:
    raise SystemExit("GATE_API_KEY/GATE_API_SECRET tidak ditemukan. Pastikan file .env ada dan variabelnya benar.")

# Lindungi dari placeholder yang tak sengaja terpakai
if API_KEY == "" or API_SECRET == "":
    raise SystemExit("Masih memakai placeholder API key/secret dari source. Isi .env dengan key milik Anda.")


HOST   = "https://api-testnet.gateapi.io" if USE_TESTNET else "https://api.gateio.ws"
PREFIX = "/api/v4"

SYMBOL            = os.getenv("SYMBOL", "BTC_USDT")
ACCOUNT           = os.getenv("GATE_ACCOUNT", "spot")  # "spot" atau "unified"

# Range configuration - support manual or auto percentage
AUTO_RANGE_PERCENT = os.getenv("AUTO_RANGE_PERCENT")  # e.g., "5" for ±5%
if AUTO_RANGE_PERCENT:
    # Auto range will be calculated in main() based on current price
    LOWER_BOUND = None
    UPPER_BOUND = None
    AUTO_RANGE_PERCENT = float(AUTO_RANGE_PERCENT)
else:
    # Manual range
    LOWER_BOUND = float(os.getenv("LOWER", "55000"))
    UPPER_BOUND = float(os.getenv("UPPER", "65000"))

NUM_GRIDS         = int(os.getenv("GRIDS", "20"))
QUOTE_PER_ORDER   = float(os.getenv("QUOTE_PER_ORDER", "50"))   # dalam quote (USDT)
USE_GEOMETRIC     = os.getenv("GEOMETRIC", "0") == "1"
POLL_INTERVAL_S   = int(os.getenv("POLL", "5"))
STATE_FILE        = os.getenv("STATE_FILE", "grid_state.json")
MAX_RETRIES       = int(os.getenv("MAX_RETRIES", "3"))  # New: retry attempts
RETRY_DELAY_S     = int(os.getenv("RETRY_DELAY", "10"))  # New: delay between retries

session = requests.Session()
session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

# Global variables for background bot control
bot_thread = None
bot_stop_flag = threading.Event()
background_mode = False  # Flag to suppress verbose output
restart_on_error = True  # New: auto-restart flag

def _sha512_hex(s: str) -> str:
    return hashlib.sha512((s or "").encode("utf-8")).hexdigest()

def _sign_headers(method: str, url_path: str, query_str: str, body_str: str):
    ts = str(int(time.time()))
    sign_str = f"{method}\n{PREFIX}{url_path}\n{query_str}\n{_sha512_hex(body_str)}\n{ts}"
    sign = hmac.new(API_SECRET.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha512).hexdigest()
    return {"KEY": API_KEY, "Timestamp": ts, "SIGN": sign}

def _req(method: str, url_path: str, params: dict | None = None, body: dict | None = None, auth: bool = False):
    """Enhanced request with retry logic"""
    max_attempts = MAX_RETRIES
    for attempt in range(1, max_attempts + 1):
        try:
            query_str = ""
            if params:
                query_str = "&".join(f"{k}={v}" for k, v in params.items())
            body_str = json.dumps(body) if body else ""
            headers = {}
            if auth:
                headers = _sign_headers(method, url_path, query_str, body_str)
            url = f"{HOST}{PREFIX}{url_path}" + (f"?{query_str}" if query_str else "")
            r = session.request(method, url, headers=headers, data=(body_str if body else None), timeout=15)
            if r.status_code >= 400:
                raise RuntimeError(f"{r.status_code} | {r.text}")
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_attempts:
                wait_time = RETRY_DELAY_S * attempt
                log_to_file(f"Connection error (attempt {attempt}/{max_attempts}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                log_to_file(f"Connection failed after {max_attempts} attempts: {e}")
                raise
        except Exception as e:
            log_to_file(f"Request error: {e}")
            raise

# ---------- Public endpoints ----------
def get_pair_detail(symbol: str):
    return _req("GET", f"/spot/currency_pairs/{symbol}")

def get_last_price(symbol: str) -> float:
    data = _req("GET", "/spot/tickers", params={"currency_pair": symbol})
    if not data:
        raise RuntimeError("tickers empty")
    return float(data[0]["last"])

# ---------- Private endpoints (auth) ----------
def list_open_orders(symbol: str):
    return _req("GET", "/spot/orders", params={"currency_pair": symbol, "status": "open"}, auth=True)

def get_order(symbol: str, order_id: str):
    return _req("GET", f"/spot/orders/{order_id}", params={"currency_pair": symbol}, auth=True)

def create_limit_post_only(symbol: str, side: str, price_str: str, amount_str: str, text: str):
    body = {
        "text": text,                         # harus prefiks "t-" untuk custom client id
        "currency_pair": symbol,
        "type": "limit",
        "account": ACCOUNT,                   # "spot" atau "unified"
        "side": side,                         # "buy" | "sell"
        "amount": amount_str,
        "price": price_str,
        "time_in_force": "poc"                # maker-only (post-only)
    }
    return _req("POST", "/spot/orders", body=body, auth=True)

def cancel_all(symbol: str):
    return _req("DELETE", "/spot/orders", params={"currency_pair": symbol, "account": ACCOUNT}, auth=True)

# ---------- Grid helpers ----------
def grid_levels(lower: float, upper: float, n: int, geometric=False):
    """Generate n grid levels between lower and upper (excluding boundaries)"""
    if geometric:
        r = (upper / lower) ** (1.0 / (n + 1))
        return [lower * (r ** i) for i in range(1, n + 1)]
    else:
        step = (upper - lower) / (n + 1)
        return [lower + i * step for i in range(1, n + 1)]

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"orders": {}}
    try:
        with open(STATE_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                log_to_file(f"⚠️ {STATE_FILE} is empty, returning default state")
                return {"orders": {}}
            return json.loads(content)
    except json.JSONDecodeError as e:
        log_to_file(f"❌ Failed to parse {STATE_FILE}: {e}. Creating backup and returning default state")
        import shutil
        backup_file = f"{STATE_FILE}.backup.{int(time.time())}"
        try:
            shutil.copy(STATE_FILE, backup_file)
            log_to_file(f"💾 Corrupted state backed up to {backup_file}")
        except Exception as backup_err:
            log_to_file(f"⚠️ Could not create backup: {backup_err}")
        return {"orders": {}}
    except Exception as e:
        log_to_file(f"Unexpected error loading state: {e}")
        return {"orders": {}}

def save_state(s):
    """Save state with atomic write and error handling"""
    try:
        temp_file = f"{STATE_FILE}.tmp"
        with open(temp_file, "w") as f:
            json.dump(s, f, indent=2)
        import shutil
        shutil.move(temp_file, STATE_FILE)
    except Exception as e:
        log_to_file(f"❌ Failed to save state: {e}")

def save_to_json(filename: str, data):
    """Save data to JSON file with timestamp"""
    try:
        output = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "data": data
        }
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
    except Exception as e:
        print(f"❌ Failed to save {filename}: {e}")



def log_to_file(message: str, filename: str = "bot.log"):
    """Enhanced logging with error handling"""
    try:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass  # Fail silently for logging errors

def quantize(val: float, decimals: int) -> str:
    q = Decimal(val).quantize(Decimal(f"1e-{decimals}"), rounding=ROUND_DOWN)
    return format(q, f".{decimals}f")

def ensure_minimums(pair, side: str, price: float, amount: float) -> float:
    """
    Gate menyediakan:
      - amount_precision (jumlah desimal untuk amount/base)
      - precision         (jumlah desimal untuk price)
      - min_base_amount
      - min_quote_amount
    Untuk BUY: pastikan amount*price >= min_quote_amount.
    Untuk SELL: pastikan amount >= min_base_amount.
    """
    min_base  = Decimal(pair.get("min_base_amount") or "0")
    min_quote = Decimal(pair.get("min_quote_amount") or "0")
    aprec = int(pair["amount_precision"])

    amt = Decimal(str(amount))
    if side == "sell" and min_base > 0 and amt < min_base:
        amt = min_base
    if side == "buy" and min_quote > 0:
        need = (min_quote / Decimal(str(price)))
        if amt < need:
            amt = need
    # bulatkan ke amount_precision
    step = Decimal(f"1e-{aprec}")
    amt = (amt // step) * step
    return float(amt)

def main():
    """Main trading loop - exceptions handled by auto-restart wrapper"""
    global LOWER_BOUND, UPPER_BOUND
    
    # Initialize
    pair = get_pair_detail(SYMBOL)
    price_prec  = int(pair["precision"])
    amount_prec = int(pair["amount_precision"])

    # harga terakhir
    last = get_last_price(SYMBOL)
    
    # ALWAYS recalculate range based on current price at startup
    # This ensures orders are placed near current price when bot restarts
    if AUTO_RANGE_PERCENT:
        LOWER_BOUND = last * (1 - AUTO_RANGE_PERCENT / 100)
        UPPER_BOUND = last * (1 + AUTO_RANGE_PERCENT / 100)
        range_msg = f"Auto range mode: ±{AUTO_RANGE_PERCENT}% from current price {last:,.2f}"
        print(range_msg)
        log_to_file(range_msg)
        print(f"Calculated range: {LOWER_BOUND:,.2f} - {UPPER_BOUND:,.2f}\n")
    else:
        # Even with manual range, recalculate based on current price for dynamic trading
        old_lower = LOWER_BOUND
        old_upper = UPPER_BOUND
        LOWER_BOUND = last * 0.95  # 5% below current
        UPPER_BOUND = last * 1.05  # 5% above current
        range_msg = f"Manual range mode adjusted to current price {last:,.2f} (was {old_lower:,.2f}-{old_upper:,.2f})"
        print(range_msg)
        log_to_file(range_msg)
        print(f"Dynamic range: {LOWER_BOUND:,.2f} - {UPPER_BOUND:,.2f}\n")

    # LOGIC KHUSUS: Awal selalu place BUY orders saja (GRIDS = jumlah buy orders)
    # Generate GRIDS levels DI BAWAH harga current untuk buy orders
    # Use narrower range for buy orders: from (current - AUTO_RANGE_PERCENT%) to current
    buy_upper_limit = last  # Buy orders up to current price
    buy_lower_limit = LOWER_BOUND  # From lower bound
    levels = grid_levels(buy_lower_limit, buy_upper_limit, NUM_GRIDS, geometric=USE_GEOMETRIC)
    sorted_levels = sorted(levels)
    
    # Setup awal: semua levels adalah BUY orders (di bawah harga current)
    buy_lvls = sorted_levels
    sell_lvls = []  # Tidak ada sell order di awal
    
    state = load_state()
    
    # Log bot start
    start_msg = f"Bot started | Symbol: {SYMBOL} | Range: {LOWER_BOUND:.2f}-{UPPER_BOUND:.2f} | Grids: {NUM_GRIDS} | Mode: {'Geometric' if USE_GEOMETRIC else 'Arithmetic'}"
    print(start_msg)
    log_to_file(start_msg)
    print(f"\nCurrent Price: {last:.2f}")
    print(f"Strategy: DCA Grid Bot - Start with {NUM_GRIDS} BUY orders")
    print(f"Logic: Buy filled → Convert to SELL | Sell filled → Place new BUY")
    print(f"       Maintain max {NUM_GRIDS} active orders at all times\n")
    
    # Helper: get available balance
    def get_available_balance(currency: str) -> float:
        try:
            # Gunakan endpoint yang sesuai dengan account type
            if ACCOUNT == "unified":
                # Untuk unified account, gunakan /unified/accounts
                accounts = _req("GET", "/unified/accounts", auth=True)
                # Response format berbeda untuk unified
                if isinstance(accounts, dict):
                    details = accounts.get('details', {})
                    for curr, info in details.items():
                        if curr.upper() == currency.upper():
                            return float(info.get('available', 0))
            else:
                # Untuk spot account, gunakan /spot/accounts
                accounts = _req("GET", "/spot/accounts", auth=True)
                for acc in accounts:
                    if acc.get('currency', '').upper() == currency.upper():
                        return float(acc.get('available', 0))
            return 0.0
        except Exception as e:
            print(f"⚠️  Error getting balance for {currency}: {e}")
            return 0.0

    # Track allocated balance to avoid double-counting
    allocated_balance = {"base": 0.0, "quote": 0.0}
    
    def place_if_missing(level: float, side: str, skip_balance_check: bool = False) -> bool:
        """Place order if missing. Returns True if placed, False if skipped."""
        lvl_key = f"{level:.10f}"
        if lvl_key in state["orders"]:
            return False
        px = quantize(level, price_prec)
        # sizing: quote tetap per grid → amount = QUOTE_PER_ORDER / price
        amt_raw = QUOTE_PER_ORDER / float(px)
        amt = ensure_minimums(pair, side, float(px), amt_raw)
        if amt <= 0:
            log_to_file(f"Skip level {px}: amount too small")
            return False
        amt_str = quantize(amt, amount_prec)
        
        # Balance check (kecuali dipaksa skip)
        if not skip_balance_check:
            base_curr, quote_curr = SYMBOL.split('_')
            if side == "buy":
                required = QUOTE_PER_ORDER
                available = get_available_balance(quote_curr) - allocated_balance["quote"]
                if available < required:
                    log_to_file(f"Skip buy@{px}: Need {required:.2f} {quote_curr}, available {available:.2f}")
                    return False
                allocated_balance["quote"] += required
            else:  # sell
                required = float(amt_str)
                available = get_available_balance(base_curr) - allocated_balance["base"]
                if available < required:
                    log_to_file(f"Skip sell@{px}: Need {required:.8f} {base_curr}, available {available:.8f}")
                    return False
                allocated_balance["base"] += required
        
        text = f"t-grid-{side}-{px}"
        try:
            od = create_limit_post_only(SYMBOL, side, px, amt_str, text)
            if od.get("status") == "open":
                state["orders"][lvl_key] = {"orderId": od["id"], "side": side, "amount": amt_str, "price": px}
                msg = f"OPEN {side} {amt_str} @ {px} (id={od['id']})"
                if not background_mode:
                    print(msg)
                log_to_file(msg)
                return True
            else:
                msg = f"Order {side}@{px} not open (finish_as={od.get('finish_as')})"
                if not background_mode:
                    print(msg)
                log_to_file(msg)
                if not skip_balance_check:
                    if side == "buy":
                        allocated_balance["quote"] -= QUOTE_PER_ORDER
                    else:
                        allocated_balance["base"] -= float(amt_str)
                return False
        except RuntimeError as e:
            error_msg = f"Create order failed: {e}"
            if not background_mode:
                print(error_msg)
            log_to_file(error_msg)
            if not skip_balance_check:
                if side == "buy":
                    allocated_balance["quote"] -= QUOTE_PER_ORDER
                else:
                    allocated_balance["base"] -= float(amt_str)
            return False

    # Cek balance awal
    base_curr, quote_curr = SYMBOL.split('_')
    print(f"\nFetching balance from Gate.io ({ACCOUNT} account)...")
    initial_base = get_available_balance(base_curr)
    initial_quote = get_available_balance(quote_curr)
    
    print(f"\n{'='*60}")
    print(f"INITIAL BALANCE CHECK")
    print(f"{'='*60}")
    print(f"Account Type: {ACCOUNT}")
    print(f"Symbol: {SYMBOL}")
    print(f"{base_curr}: {initial_base:.8f}")
    print(f"{quote_curr}: {initial_quote:.2f}")
    print(f"\nRequired per buy order: {QUOTE_PER_ORDER:.2f} {quote_curr}")
    print(f"Max buy orders possible: {int(initial_quote / QUOTE_PER_ORDER) if QUOTE_PER_ORDER > 0 else 0}")
    print(f"{'='*60}")
    
    if initial_base < 0.0001:  # hampir tidak punya base currency
        print(f"\n✅ Starting DCA Grid Bot with {quote_curr}-only:")
        print(f"   → Placing {NUM_GRIDS} BUY orders below current price")
        print(f"   → Buy filled → Auto place SELL order (+2% profit target)")
        print(f"   → Sell filled → Auto place new BUY (dynamic range)")
        print(f"   → Total {NUM_GRIDS} orders maintained at all times\n")
        log_to_file(f"Starting DCA Grid Bot: {initial_quote:.2f} {quote_curr}, {initial_base:.8f} {base_curr}")
    else:
        print(f"\n✓ Have both {base_curr} and {quote_curr} balance")
        print(f"  → Starting DCA Grid Bot with {NUM_GRIDS} BUY orders")
        print(f"  → Will maintain {NUM_GRIDS} active orders (buy + sell mix)\n")
        log_to_file(f"Starting with dual balance: {initial_quote:.2f} {quote_curr}, {initial_base:.8f} {base_curr}")
    
    # Preview grid levels (show from top to bottom - closest to current price first)
    print("\n" + "="*60)
    print("INITIAL BUY ORDERS SETUP")
    print("="*60)
    print(f"Current Price: {last:,.2f}\n")
    print(f"Placing {len(buy_lvls)} BUY orders below current price:")
    # Reverse order untuk display: level terdekat current price (paling atas) ditampilkan pertama
    for i, p in enumerate(reversed(buy_lvls), 1):
        distance_pct = ((last - p) / last) * 100
        print(f"  Grid {i}. Buy @ {p:,.2f} ({distance_pct:.2f}% below current)")
    print("\nNo SELL orders placed initially.")
    print("SELL orders will be created when BUY orders are filled.")
    print("="*60 + "\n")
    
    # Check existing orders from state (if restarting after crash)
    existing_orders = len(state["orders"])
    existing_buys = sum(1 for o in state["orders"].values() if o.get("side") == "buy")
    existing_sells = sum(1 for o in state["orders"].values() if o.get("side") == "sell")
    
    if existing_orders > 0:
        print(f"\n🔄 RESTART DETECTED - Loading existing orders from state file")
        print(f"   Found: {existing_buys} buy orders, {existing_sells} sell orders (Total: {existing_orders})")
        log_to_file(f"Restart: Loaded {existing_orders} existing orders ({existing_buys}B/{existing_sells}S)")
        
        # Only place orders if we need more to reach NUM_GRIDS
        orders_needed = NUM_GRIDS - existing_orders
        if orders_needed > 0:
            print(f"   Need {orders_needed} more orders to reach target of {NUM_GRIDS}")
            print(f"   Placing {orders_needed} new BUY orders...\n")
            log_to_file(f"Placing {orders_needed} additional buy orders to reach {NUM_GRIDS} target")
            
            # Place only the needed amount of new buy orders
            placed_count = 0
            for p in buy_lvls:
                if placed_count >= orders_needed:
                    break
                if place_if_missing(p, "buy"):
                    placed_count += 1
        else:
            print(f"   ✅ Already have {existing_orders} orders (target: {NUM_GRIDS})")
            print(f"   Skipping initial order placement, continuing with existing orders\n")
            log_to_file(f"Restart: Already at target, skipping initial placement")
    else:
        # Fresh start - place all initial orders
        print("\n🆕 FRESH START - No existing orders found")
        if not background_mode:
            print("Placing BUY orders, please wait...\n")
        
        for p in buy_lvls:
            place_if_missing(p, "buy")
    
    # TIDAK place sell orders di awal (sell akan dibuat saat buy filled)
    # for p in sell_lvls:  # Commented out
    #     place_if_missing(p, "sell")
    
    # Save state setelah semua order di-place
    save_state(state)
    
    # Reset allocated balance karena balance sudah ter-commit di exchange
    allocated_balance["base"] = 0.0
    allocated_balance["quote"] = 0.0
    
    # Tunggu sebentar untuk memastikan semua order ter-sync
    time.sleep(1)
    
    # Hitung yang berhasil di-place dari state yang sudah di-save
    placed_buys = sum(1 for o in state["orders"].values() if o.get("side") == "buy")
    placed_sells = sum(1 for o in state["orders"].values() if o.get("side") == "sell")
    
    summary_msg = f"Initial orders placed: {placed_buys} buys, {placed_sells} sells"
    print(f"\n{summary_msg}\n")
    log_to_file(summary_msg)
    
    print("="*60)
    print("✅ Bot is now running in background")
    print("💡 You can return to menu and bot will keep monitoring")
    print("📊 Use 'Check Bot Status' to see activity")
    print("🛑 Use 'Stop Background Bot' to stop safely")
    print("="*60 + "\n")

    # MAIN LOOP - Enhanced with error handling
    loop_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while not bot_stop_flag.is_set():
        try:
            # Get current price
            last = get_last_price(SYMBOL)
            
            # Reset error counter on success
            consecutive_errors = 0
            
            # Status update
            loop_count += 1
            if loop_count % 12 == 0:
                open_count = len(state["orders"])
                buy_count = sum(1 for o in state["orders"].values() if o["side"] == "buy")
                sell_count = open_count - buy_count
                status_msg = f"[{time.strftime('%H:%M:%S')}] Price: {last:,.2f} | Orders: {buy_count}B/{sell_count}S"
                print(status_msg)
                log_to_file(status_msg)
                
        except Exception as e:
            consecutive_errors += 1
            error_msg = f"Ticker error ({consecutive_errors}/{max_consecutive_errors}): {e}"
            print(error_msg)
            log_to_file(error_msg)
            
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(f"Too many consecutive errors ({consecutive_errors}). Restarting...")
            
            time.sleep(POLL_INTERVAL_S)
            continue

        # Risk control
        try:
            if last < LOWER_BOUND * 0.995 or last > UPPER_BOUND * 1.005:
                exit_msg = f"Price {last} keluar dari [{LOWER_BOUND}, {UPPER_BOUND}] → CANCEL ALL & EXIT"
                print(exit_msg)
                log_to_file(exit_msg)
                try:
                    cancel_all(SYMBOL)
                    log_to_file("All orders cancelled successfully")
                except Exception as e:
                    log_to_file(f"Cancel all gagal: {e}")
                break
        except Exception as e:
            log_to_file(f"Risk control check error: {e}")

        # Check open orders
        try:
            open_ods = list_open_orders(SYMBOL)
        except Exception as e:
            error_msg = f"List open error: {e}"
            print(error_msg)
            log_to_file(error_msg)
            time.sleep(POLL_INTERVAL_S)
            continue

        # Process filled orders
        try:
            open_ids = {o["id"] for o in open_ods}
            for lvl_key, rec in list(state["orders"].items()):
                oid = rec["orderId"]
                if oid in open_ids:
                    continue
                
                try:
                    od = get_order(SYMBOL, oid)
                    status = od.get("status")
                except Exception:
                    status = "closed"
                
                level = float(lvl_key)
                if status in ("closed", "cancelled"):
                    side = rec["side"]
                    fill_msg = f"Order FILLED: {side} @ {rec.get('price')} (id={oid})"
                    print(fill_msg)
                    log_to_file(fill_msg)
                    
                    if side == "buy":
                        sell_target = level * 1.02
                        current_price = get_last_price(SYMBOL)
                        if sell_target < current_price:
                            sell_target = current_price * 1.01
                        
                        profit_pct = ((sell_target / level - 1) * 100)
                        mirror_msg = f"✓ BUY filled @ {level:.2f} → placing SELL @ {sell_target:.2f} (+{profit_pct:.1f}% profit target)"
                        print(mirror_msg)
                        log_to_file(mirror_msg)
                        place_if_missing(sell_target, "sell", skip_balance_check=True)
                        
                    else:
                        current_price = get_last_price(SYMBOL)
                        buy_percentage = AUTO_RANGE_PERCENT if AUTO_RANGE_PERCENT else 2.0
                        new_buy_level = current_price * (1 - buy_percentage / 100)
                        
                        mirror_msg = f"✓ SELL filled @ {level:.2f} → placing BUY @ {new_buy_level:.2f} (current {current_price:.2f} - {buy_percentage}%)"
                        print(mirror_msg)
                        log_to_file(mirror_msg)
                        place_if_missing(new_buy_level, "buy", skip_balance_check=True)
                    
                    state["orders"].pop(lvl_key, None)
                    save_state(state)
        except Exception as e:
            log_to_file(f"Order processing error: {e}")

        # Maintenance
        try:
            total_orders = len(state["orders"])
            if total_orders < NUM_GRIDS:
                orders_needed = NUM_GRIDS - total_orders
                current_price = get_last_price(SYMBOL)
                new_lower = current_price * (1 - AUTO_RANGE_PERCENT / 100) if AUTO_RANGE_PERCENT else LOWER_BOUND
                new_buy_levels = grid_levels(new_lower, current_price, orders_needed, geometric=USE_GEOMETRIC)
                
                for new_level in new_buy_levels:
                    lvl_key = f"{new_level:.10f}"
                    if lvl_key not in state["orders"]:
                        place_if_missing(new_level, "buy")
        except Exception as e:
            log_to_file(f"Maintenance error: {e}")

        save_state(state)
        time.sleep(POLL_INTERVAL_S)
    
    stop_msg = "🛑 Bot stopped gracefully"
    print(stop_msg)
    log_to_file(stop_msg)

def main_with_auto_restart():
    """Wrapper with automatic restart on error"""
    restart_count = 0
    max_restarts = 10
    restart_delay = 30  # seconds
    
    while restart_on_error and not bot_stop_flag.is_set():
        try:
            main()
            # If main() exits normally, break
            break
        except Exception as e:
            restart_count += 1
            error_msg = f"Bot crashed (restart #{restart_count}/{max_restarts}): {e}"
            print(f"\n❌ {error_msg}")
            log_to_file(error_msg)
            
            # Log traceback
            import traceback
            traceback_str = traceback.format_exc()
            log_to_file(f"Traceback:\n{traceback_str}")
            
            if restart_count >= max_restarts:
                final_msg = f"Maximum restart attempts ({max_restarts}) reached. Stopping bot."
                print(f"\n⛔ {final_msg}")
                log_to_file(final_msg)
                break
            
            if not bot_stop_flag.is_set():
                restart_msg = f"Restarting bot in {restart_delay} seconds..."
                print(f"\n🔄 {restart_msg}")
                log_to_file(restart_msg)
                
                for i in range(restart_delay):
                    if bot_stop_flag.is_set():
                        print("\n🛑 Restart cancelled by user")
                        return
                    time.sleep(1)
                
                print("\n♻️ Restarting bot now...")
                log_to_file("Bot restarting...")

def start_bot_background():
    """Start bot in background thread with auto-restart"""
    global bot_thread, bot_stop_flag, background_mode, restart_on_error
    
    if bot_thread and bot_thread.is_alive():
        print("\n⚠️  Bot is already running in background!")
        print("   Use 'Stop Background Bot' first if you want to restart.\n")
        return
    
    bot_stop_flag.clear()
    background_mode = True
    restart_on_error = True
    
    bot_thread = threading.Thread(target=main_with_auto_restart, daemon=True)
    bot_thread.start()
    
    time.sleep(6)
    
    if not bot_thread.is_alive():
        print("\n❌ Bot failed to start. Check bot.log for errors.\n")
        return
    
    print("\n✅ Bot initialized and running in background with auto-restart enabled.")
    print("💡 Bot will automatically restart if it encounters errors.")
    print("📊 Use 'Check Bot Status' to monitor activity.")
    print("🛑 Use 'Stop Background Bot' to stop safely.\n")

def stop_bot_background():
    """Stop background bot safely"""
    global bot_thread, bot_stop_flag, restart_on_error
    
    if not bot_thread or not bot_thread.is_alive():
        print("\n⚠️  No bot is running in background.\n")
        return
    
    print("\n🛑 Stopping bot... please wait...")
    restart_on_error = False  # Disable auto-restart
    bot_stop_flag.set()
    bot_thread.join(timeout=10)
    
    if bot_thread.is_alive():
        print("⚠️  Bot did not stop gracefully (still running)")
    else:
        print("✅ Bot stopped successfully\n")
        bot_thread = None

def check_bot_status():
    """Check if bot is running and show status"""
    global bot_thread
    
    print("\n" + "="*60)
    print(f"{'BOT STATUS':^60}")
    print("="*60)
    
    if bot_thread and bot_thread.is_alive():
        print("Status: 🟢 RUNNING in background")
        print(f"Symbol: {SYMBOL}")
        print("Auto-Restart: ✅ ENABLED")
        
        if LOWER_BOUND and UPPER_BOUND:
            print(f"Range: {LOWER_BOUND:,.0f} - {UPPER_BOUND:,.0f}")
        elif AUTO_RANGE_PERCENT:
            print(f"Range: AUTO ±{AUTO_RANGE_PERCENT}% (check log for calculated range)")
        
        # Show recent log entries
        try:
            with open("bot.log", "r", encoding="utf-8") as f:
                lines = f.readlines()
                recent = lines[-5:] if len(lines) >= 5 else lines
                print("\nRecent Activity:")
                print("-" * 60)
                for line in recent:
                    print(line.strip())
        except FileNotFoundError:
            print("\n⚠️  No log file found yet")
        except Exception as e:
            print(f"\n⚠️  Could not read log file: {e}")
    else:
        print("Status: 🔴 NOT RUNNING")
        print("Use 'Start Grid Bot' to begin trading")
    
    print("="*60 + "\n")

def auth_smoke_test():
    try:
        list_open_orders(SYMBOL)   # endpoint privat sederhana
        print("Auth OK: berhasil memanggil endpoint privat.")
    except Exception as e:
        raise SystemExit(f"Auth gagal: {e}")

# ---------- Utility Functions ----------
def check_orders_status():
    """Display all open orders with details"""
    print("\n" + "="*80)
    print(f"{'OPEN ORDERS STATUS':^80}")
    print("="*80)
    try:
        orders = list_open_orders(SYMBOL)
        
        # Auto-save to JSON
        save_to_json("open_orders.json", orders)
        
        if not orders:
            print("No open orders found.")
            return
        
        print(f"\nTotal Open Orders: {len(orders)}")
        print(f"{'ID':<15} {'Side':<6} {'Price':<12} {'Amount':<12} {'Filled':<12} {'Status':<10}")
        print("-"*80)
        
        for order in orders:
            oid = order.get('id', 'N/A')
            side = order.get('side', 'N/A').upper()
            price = order.get('price', '0')
            amount = order.get('amount', '0')
            filled = order.get('filled_amount', '0')
            status = order.get('status', 'N/A')
            
            print(f"{oid:<15} {side:<6} {price:<12} {amount:<12} {filled:<12} {status:<10}")
        
        print("-"*80)
        print(f"\n✅ Data saved to: open_orders.json")
    except Exception as e:
        print(f"Error fetching orders: {e}")

def check_balance():
    """Display spot account balance"""
    print("\n" + "="*80)
    print(f"{'SPOT ACCOUNT BALANCE':^80}")
    print("="*80)
    try:
        url_path = "/spot/accounts"
        accounts = _req("GET", url_path, auth=True)
        
        # Auto-save to JSON
        save_to_json("balance.json", accounts)
        
        if not accounts:
            print("No balances found.")
            return
        
        print(f"\n{'Currency':<10} {'Available':<20} {'Locked':<20} {'Total':<20}")
        print("-"*80)
        
        total_value_usdt = 0.0
        for acc in accounts:
            currency = acc.get('currency', 'N/A')
            available = float(acc.get('available', 0))
            locked = float(acc.get('locked', 0))
            total = available + locked
            
            if total > 0:
                print(f"{currency:<10} {available:<20.8f} {locked:<20.8f} {total:<20.8f}")
                
                # Estimate USDT value for major coins
                if currency == 'USDT':
                    total_value_usdt += total
                elif currency in ['BTC', 'ETH'] and total > 0:
                    try:
                        pair = f"{currency}_USDT"
                        price = get_last_price(pair)
                        total_value_usdt += total * price
                    except:
                        pass
        
        print("-"*80)
        print(f"Estimated Total Value: ~{total_value_usdt:.2f} USDT")
        print("="*80)
        print(f"\n✅ Data saved to: balance.json")
    except Exception as e:
        print(f"Error fetching balance: {e}")
        print(f"Estimated Total Value: ~{total_value_usdt:.2f} USDT")
        print("="*80)
    except Exception as e:
        print(f"Error fetching balance: {e}")

def emergency_stop():
    """Cancel all open orders immediately and stop bot"""
    global bot_thread, bot_stop_flag, restart_on_error
    
    print("\n" + "="*80)
    print(f"{'EMERGENCY STOP - CANCEL ALL ORDERS':^80}")
    print("="*80)
    
    confirm = input("\n⚠️  WARNING: This will cancel ALL open orders and STOP the bot! Continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    try:
        # Step 1: Disable auto-restart to prevent bot from restarting
        print("\n🛑 Disabling auto-restart...")
        restart_on_error = False
        
        # Step 2: Stop background bot if running
        if bot_thread and bot_thread.is_alive():
            print("🛑 Stopping background bot...")
            bot_stop_flag.set()
            bot_thread.join(timeout=5)
            if bot_thread.is_alive():
                print("⚠️  Bot still running (may take a moment to stop)")
            else:
                print("✅ Bot stopped successfully")
                bot_thread = None
        
        # Step 3: Cancel all orders
        print(f"\n🗑️  Cancelling all orders for {SYMBOL}...")
        result = cancel_all(SYMBOL)
        
        if isinstance(result, list):
            print(f"\n✅ Successfully cancelled {len(result)} orders")
            for order in result:
                print(f"   - Order ID: {order.get('id')} | Status: {order.get('status')}")
        else:
            print(f"\n✅ All orders cancelled successfully")
        
        # Step 4: Clear state file
        state = load_state()
        state["orders"] = {}
        save_state(state)
        print("✅ State file cleared")
        log_to_file(f"Emergency stop executed - all orders cancelled, bot stopped for {SYMBOL}")
        
        print("\n" + "="*80)
        print("✅ EMERGENCY STOP COMPLETE")
        print("="*80)
        print("\n💡 Bot is now stopped and will NOT auto-restart.")
        print("   To start trading again, use option 1 from the menu.")
        
    except Exception as e:
        print(f"\n❌ Error during emergency stop: {e}")
        log_to_file(f"Emergency stop error: {e}")
        print("="*80)

def live_monitor():
    """Live monitoring dashboard with auto-refresh"""
    import os
    
    def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n🔴 Starting Live Monitor (Press Ctrl+C to exit)...")
    time.sleep(2)
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("="*100)
            print(f"{'GATE.IO GRID BOT - LIVE MONITOR':^100}")
            print(f"{'Symbol: ' + SYMBOL + ' | Refresh: ' + str(POLL_INTERVAL_S) + 's':^100}")
            print("="*100)
            
            # Current Price
            try:
                last_price = get_last_price(SYMBOL)
                price_status = "🟢 NORMAL"
                if LOWER_BOUND and UPPER_BOUND:
                    if last_price < LOWER_BOUND * 0.995:
                        price_status = "🔴 BELOW RANGE"
                    elif last_price > UPPER_BOUND * 1.005:
                        price_status = "🔴 ABOVE RANGE"
                
                print(f"\n💰 Current Price: {last_price:,.2f} USDT | Status: {price_status}")
                
                if LOWER_BOUND and UPPER_BOUND:
                    print(f"📊 Grid Range: {LOWER_BOUND:,.2f} - {UPPER_BOUND:,.2f} USDT")
                elif AUTO_RANGE_PERCENT:
                    print(f"📊 Grid Range: AUTO ±{AUTO_RANGE_PERCENT}% (start bot to see calculated range)")
            except Exception as e:
                print(f"\n❌ Price fetch error: {e}")
                last_price = 0
            
            # Balance Summary
            print("\n" + "-"*100)
            print("💼 BALANCE SUMMARY")
            print("-"*100)
            try:
                accounts = _req("GET", "/spot/accounts", auth=True)
                for acc in accounts:
                    available = float(acc.get('available', 0))
                    locked = float(acc.get('locked', 0))
                    if available > 0 or locked > 0:
                        currency = acc['currency']
                        total = available + locked
                        print(f"   {currency:<8} | Available: {available:>15.8f} | Locked: {locked:>15.8f} | Total: {total:>15.8f}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Open Orders
            print("\n" + "-"*100)
            print("📋 OPEN ORDERS")
            print("-"*100)
            try:
                orders = list_open_orders(SYMBOL)
                buy_orders = [o for o in orders if o['side'] == 'buy']
                sell_orders = [o for o in orders if o['side'] == 'sell']
                
                print(f"   Total: {len(orders)} orders | 🟢 Buy: {len(buy_orders)} | 🔴 Sell: {len(sell_orders)}")
                
                if orders:
                    print(f"\n   {'ID':<15} {'Side':<6} {'Price':<12} {'Amount':<12} {'Filled':<10} {'Status':<10}")
                    for order in orders[:10]:  # Show first 10 orders
                        side_icon = "🟢" if order['side'] == 'buy' else "🔴"
                        print(f"   {order['id']:<15} {side_icon} {order['side']:<4} {float(order['price']):>11.2f} "
                              f"{order['amount']:<12} {order.get('filled_amount', '0'):<10} {order['status']:<10}")
                    
                    if len(orders) > 10:
                        print(f"   ... and {len(orders) - 10} more orders")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Grid State
            print("\n" + "-"*100)
            print("🔧 GRID STATE")
            print("-"*100)
            state = load_state()
            state_orders = state.get("orders", {})
            print(f"   Tracked Orders in State File: {len(state_orders)}")
            
            if state_orders:
                buy_count = sum(1 for o in state_orders.values() if o.get('side') == 'buy')
                sell_count = sum(1 for o in state_orders.values() if o.get('side') == 'sell')
                print(f"   🟢 Buy: {buy_count} | 🔴 Sell: {sell_count}")
            
            # Footer
            print("\n" + "="*100)
            print(f"{'Last Update: ' + time.strftime('%Y-%m-%d %H:%M:%S'):^100}")
            print(f"{'Press Ctrl+C to return to menu (bot will keep running)':^100}")
            print("="*100)
            
            time.sleep(POLL_INTERVAL_S)
            
    except KeyboardInterrupt:
        print("\n\n✅ Live monitor stopped. Returning to menu...")
        print("💡 Background bot is still running (if started)\n")
        time.sleep(2)
        return

def profit_calculator():
    """Calculate estimated profit from filled orders"""
    print("\n" + "="*80)
    print(f"{'PROFIT CALCULATOR (BETA)':^80}")
    print("="*80)
    
    try:
        # Fetch closed orders (last 100)
        url_path = "/spot/my_trades"
        params = {"currency_pair": SYMBOL, "limit": 100}
        trades = _req("GET", url_path, params=params, auth=True)
        
        if not trades:
            print("\nNo trade history found.")
            return
        
        total_buy_usdt = 0
        total_sell_usdt = 0
        total_fee_usdt = 0
        buy_count = 0
        sell_count = 0
        
        print(f"\nAnalyzing last {len(trades)} trades...")
        print(f"\n{'Time':<20} {'Side':<6} {'Price':<12} {'Amount':<12} {'Total':<12} {'Fee':<10}")
        print("-"*80)
        
        for trade in trades:
            side = trade.get('side', 'N/A')
            price = float(trade.get('price', 0))
            amount = float(trade.get('amount', 0))
            total = price * amount
            fee = float(trade.get('fee', 0))
            trade_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(trade.get('create_time', 0))))
            
            print(f"{trade_time:<20} {side.upper():<6} {price:<12.2f} {amount:<12.8f} {total:<12.2f} {fee:<10.8f}")
            
            if side == 'buy':
                total_buy_usdt += total
                buy_count += 1
            else:
                total_sell_usdt += total
                sell_count += 1
            
            total_fee_usdt += fee * price if trade.get('fee_currency') != 'USDT' else fee
        
        print("-"*80)
        print(f"\nSummary:")
        print(f"  Total Buy:  {total_buy_usdt:.2f} USDT ({buy_count} trades)")
        print(f"  Total Sell: {total_sell_usdt:.2f} USDT ({sell_count} trades)")
        print(f"  Total Fees: {total_fee_usdt:.4f} USDT")
        print(f"  Net P&L:    {(total_sell_usdt - total_buy_usdt - total_fee_usdt):.2f} USDT")
        
        if total_buy_usdt > 0:
            roi = ((total_sell_usdt - total_buy_usdt - total_fee_usdt) / total_buy_usdt) * 100
            print(f"  ROI:        {roi:.2f}%")
        
        print("="*80)
    except Exception as e:
        print(f"Error calculating profit: {e}")

def interactive_menu():
    """Main interactive menu"""
    while True:
        print("\n" + "="*80)
        print(f"{'GATE.IO GRID BOT v2.1 - CONTROL PANEL':^80}")
        print("="*80)
        print(f"\nMode: {'🧪 TESTNET' if USE_TESTNET else '🚀 MAINNET (LIVE)'}")
        print(f"Symbol: {SYMBOL}")
        print(f"Account: {ACCOUNT}")
        
        if AUTO_RANGE_PERCENT:
            print(f"Grid Range: AUTO (±{AUTO_RANGE_PERCENT}% from current price)")
        else:
            print(f"Grid Range: {LOWER_BOUND:,.0f} - {UPPER_BOUND:,.0f} USDT")
        
        print(f"Grid Levels: {NUM_GRIDS} | Quote per Order: {QUOTE_PER_ORDER} USDT")
        print(f"Auto-Restart: ✅ ENABLED | Max Retries: {MAX_RETRIES}")
        print("\n" + "-"*80)
        print("📋 MENU OPTIONS:")
        print("-"*80)
        print("  1. 🚀 Start Grid Bot (Background with Auto-Restart)")
        print("  2. 📊 Check Bot Status")
        print("  3. 🛑 Stop Background Bot")
        print("  4. 📈 Live Monitor Dashboard")
        print("  5. 📋 Check Open Orders")
        print("  6. 💰 Check Account Balance")
        print("  7. 💵 Profit Calculator")
        print("  8. ⛔ Emergency Stop (Cancel All Orders)")
        print("  9. 🔧 Auth Test")
        print("  0. ❌ Exit")
        print("="*80)
        
        choice = input("\nSelect option (0-9): ").strip()
        
        if choice == '1':
            print("\n🚀 Starting Grid Bot with Auto-Restart...\n")
            try:
                auth_smoke_test()
                start_bot_background()
            except Exception as e:
                print(f"\n❌ Failed to start bot: {e}")
            input("Press Enter to continue...")
        
        elif choice == '2':
            check_bot_status()
            input("Press Enter to continue...")
        
        elif choice == '3':
            stop_bot_background()
            input("Press Enter to continue...")
        
        elif choice == '4':
            live_monitor()
        
        elif choice == '5':
            check_orders_status()
            input("\nPress Enter to return to menu...")
        
        elif choice == '6':
            check_balance()
            input("\nPress Enter to return to menu...")
        
        elif choice == '7':
            profit_calculator()
            input("\nPress Enter to return to menu...")
        
        elif choice == '8':
            emergency_stop()
            input("\nPress Enter to return to menu...")
        
        elif choice == '9':
            print("\n🔐 Testing authentication...")
            try:
                auth_smoke_test()
            except Exception as e:
                print(f"❌ Auth failed: {e}")
            input("\nPress Enter to return to menu...")
        
        elif choice == '0':
            # Stop bot before exiting
            if bot_thread and bot_thread.is_alive():
                print("\n🛑 Stopping background bot before exit...")
                stop_bot_background()
            print("\n👋 Goodbye! Happy trading!")
            break
        
        else:
            print("\n❌ Invalid option. Please select 0-9.")
            time.sleep(1)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'start':
            print("🚀 Starting Grid Bot in CLI mode with auto-restart...")
            auth_smoke_test()
            main_with_auto_restart()
        elif command == 'monitor':
            print("📊 Starting Live Monitor...")
            live_monitor()
        elif command == 'orders':
            check_orders_status()
        elif command == 'balance':
            check_balance()
        elif command == 'profit':
            profit_calculator()
        elif command == 'stop':
            emergency_stop()
        elif command == 'test':
            auth_smoke_test()
        else:
            print("❌ Unknown command. Available: start, monitor, orders, balance, profit, stop, test")
    else:
        interactive_menu()
