from connectors.gate_io import initialize_client, get_account_balance
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get trading mode from .env (default to testnet for safety)
mode = os.getenv('TRADING_MODE', 'testnet').lower()

print(f"Mode: {mode.upper()}")
print(f"Pair: {os.getenv('TRADING_PAIR', 'SOL_USDT')}")
print("-" * 50)

# Initialize client with mode from .env
client = initialize_client(mode)

# Check balances
usdt = get_account_balance(client, 'USDT')
sol = get_account_balance(client, 'SOL')

print(f"\n{mode.upper()} Balances:")
print(f"USDT: Available={usdt['available']:.2f}, Locked={usdt['locked']:.2f}")
print(f"SOL:  Available={sol['available']:.6f}, Locked={sol['locked']:.6f}")

# Calculate totals and equity
total_usdt = usdt['available'] + usdt['locked']
total_sol = sol['available'] + sol['locked']
sol_price = 124.06  # Approximate, would need to fetch real-time price

print(f"\nTotals:")
print(f"USDT Total: ${total_usdt:.2f} (Available + Locked)")
print(f"SOL Total: {total_sol:.6f} (Available + Locked)")
print(f"SOL Value: ${total_sol * sol_price:.2f} (@ ${sol_price})")
print(f"Total Equity: ${total_usdt + (total_sol * sol_price):.2f}")

print(f"\nTrading Capacity:")
print(f"Can place {int(usdt['available'] / (0.044 * sol_price))} buy orders of 0.044 SOL (~$5.50 each)")
print(f"Can place {int(sol['available'] / 0.044)} sell orders of 0.044 SOL (~$5.50 each)")
