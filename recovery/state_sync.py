"""
State Synchronization and Recovery Module

This module handles bot state persistence and recovery across restarts.
It saves critical bot state to disk and can recover from unexpected shutdowns
by reconciling saved state with actual exchange state.

Key Features:
    - Persistent state storage (JSON)
    - State/exchange reconciliation
    - Automatic recovery on restart
    - Mismatch detection and correction
    - Grid rebuild capability

Usage:
    state_manager = StateManager('bot_state.json')
    state_manager.save_state(balances, orders, grid_params, equity)
    state = state_manager.load_state(connector, pair)
"""

import json
import logging
import os
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from pathlib import Path

# Import Gate.io connector functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.gate_io import GateIOConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages bot state persistence and recovery.
    
    This class handles saving/loading bot state to/from disk and provides
    recovery mechanisms to reconcile saved state with actual exchange state.
    
    Attributes:
        state_file (str): Path to state file
        backup_dir (str): Directory for state backups
        current_state (Dict): Current loaded state
    """
    
    def __init__(self, state_file: str = 'bot_state.json', backup_dir: str = 'backups'):
        """
        Initialize state manager.
        
        Args:
            state_file: Path to state file (default: 'bot_state.json')
            backup_dir: Directory for backups (default: 'backups')
            
        Example:
            >>> manager = StateManager('bot_state.json')
            >>> manager.save_state(balances, orders, grid_params, 10000)
        """
        self.state_file = state_file
        self.backup_dir = backup_dir
        self.current_state: Optional[Dict] = None
        
        # Create backup directory if it doesn't exist
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"StateManager initialized with state file: {state_file}")
    
    def save_state(
        self,
        balances: Dict[str, float],
        active_orders: List[Dict[str, Any]],
        grid_parameters: Dict[str, Any],
        last_equity: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save current bot state to disk.
        
        This function creates a complete snapshot of the bot's state including
        balances, orders, grid configuration, and equity value.
        
        Args:
            balances: Dictionary of currency balances (e.g., {'USDT': 1000, 'SOL': 5})
            active_orders: List of active order dictionaries with keys:
                           'order_id', 'side', 'price', 'amount', 'pair'
            grid_parameters: Grid configuration dictionary with keys:
                            'pair', 'buy_levels', 'sell_levels', 'base_amount',
                            'grid_range_pct', 'total_levels', 'trend_bias'
            last_equity: Current total equity value
            metadata: Optional additional metadata
            
        Returns:
            bool: True if save successful, False otherwise
            
        Example:
            >>> balances = {'USDT': 9500.0, 'SOL': 1.5}
            >>> orders = [
            ...     {'order_id': '123', 'side': 'buy', 'price': 98.0, 'amount': 0.1},
            ...     {'order_id': '124', 'side': 'sell', 'price': 102.0, 'amount': 0.1}
            ... ]
            >>> grid_params = {
            ...     'pair': 'SOL_USDT',
            ...     'buy_levels': [98, 96, 94],
            ...     'sell_levels': [102, 104, 106]
            ... }
            >>> manager.save_state(balances, orders, grid_params, 10000.0)
        """
        try:
            # Build state dictionary
            state = {
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'balances': balances,
                'active_orders': active_orders,
                'grid_parameters': grid_parameters,
                'last_equity': last_equity,
                'metadata': metadata or {}
            }
            
            # Backup existing state file if it exists
            if os.path.exists(self.state_file):
                self._backup_state()
            
            # Write state to file
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.current_state = state
            
            logger.info(
                f"✓ State saved: {len(active_orders)} orders, "
                f"equity=${last_equity:.2f}, balances={list(balances.keys())}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(
        self,
        connector: Optional[GateIOConnector] = None,
        pair: Optional[str] = None,
        reconcile: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load bot state from disk and optionally reconcile with exchange.
        
        This function loads the saved state and can perform reconciliation
        with the actual exchange state to detect and fix mismatches.
        
        Args:
            connector: Gate.io connector for reconciliation (optional)
            pair: Trading pair for reconciliation (optional)
            reconcile: Whether to reconcile with exchange state (default: True)
            
        Returns:
            Dict with loaded state or None if load fails. State includes:
                - 'version': State file version
                - 'timestamp': When state was saved
                - 'balances': Currency balances
                - 'active_orders': List of active orders
                - 'grid_parameters': Grid configuration
                - 'last_equity': Last equity value
                - 'reconciliation': Reconciliation results (if performed)
                
        Example:
            >>> state = manager.load_state(client, 'SOL_USDT', reconcile=True)
            >>> if state:
            ...     print(f"Loaded {len(state['active_orders'])} orders")
            ...     print(f"Last equity: ${state['last_equity']:.2f}")
        """
        try:
            if not os.path.exists(self.state_file):
                logger.warning(f"State file not found: {self.state_file}")
                return None
            
            # Load state from file
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            logger.info(
                f"✓ State loaded from {state['timestamp']}: "
                f"{len(state['active_orders'])} orders, equity=${state['last_equity']:.2f}"
            )
            
            self.current_state = state
            
            # Perform reconciliation if requested
            if reconcile and connector is not None and pair is not None:
                logger.info("Starting state reconciliation with exchange...")
                reconciliation = self._reconcile_state(connector, pair, state)
                state['reconciliation'] = reconciliation
            
            return state
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse state file: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def _reconcile_state(
        self,
        connector: GateIOConnector,
        pair: str,
        saved_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reconcile saved state with actual exchange state.
        
        This internal method compares saved order IDs with actual open orders
        on the exchange and identifies mismatches.
        
        Args:
            connector: Gate.io connector
            pair: Trading pair
            saved_state: Loaded state dictionary
            
        Returns:
            Dict with reconciliation results
        """
        logger.info("=" * 60)
        logger.info("STATE RECONCILIATION")
        logger.info("=" * 60)
        
        reconciliation = {
            'timestamp': datetime.now().isoformat(),
            'expected_orders': [],
            'actual_orders': [],
            'matched_orders': [],
            'missing_orders': [],
            'unexpected_orders': [],
            'actions_taken': []
        }
        
        try:
            # Extract expected order IDs from saved state
            expected_orders = saved_state.get('active_orders', [])
            # Handle both 'order_id' and 'id' keys for backward compatibility
            expected_order_ids = {
                order.get('order_id') or order.get('id') 
                for order in expected_orders
            }
            expected_order_ids.discard(None)  # Remove any None values
            
            reconciliation['expected_orders'] = [
                {
                    'order_id': order.get('order_id') or order.get('id'),
                    'side': order.get('side', 'unknown'),
                    'price': order.get('price', 0)
                }
                for order in expected_orders
                if order.get('order_id') or order.get('id')
            ]
            
            logger.info(f"Expected {len(expected_order_ids)} orders from saved state")
            
            # Fetch actual open orders from exchange
            def _get_open_orders():
                return connector._request(
                    "GET",
                    "/spot/orders",
                    params={
                        "currency_pair": pair,
                        "status": "open"
                    },
                    auth=True
                )
            
            actual_orders = connector._retry_on_error(_get_open_orders)
            actual_order_ids = {order['id'] for order in actual_orders}
            
            reconciliation['actual_orders'] = [
                {
                    'order_id': order['id'],
                    'side': order['side'],
                    'price': float(order['price'])
                }
                for order in actual_orders
            ]
            
            logger.info(f"Found {len(actual_order_ids)} actual open orders on exchange")
            
            # Find matched orders
            matched = expected_order_ids & actual_order_ids
            reconciliation['matched_orders'] = list(matched)
            
            # Find missing orders (expected but not found)
            missing = expected_order_ids - actual_order_ids
            reconciliation['missing_orders'] = list(missing)
            
            # Find unexpected orders (found but not expected)
            unexpected = actual_order_ids - expected_order_ids
            reconciliation['unexpected_orders'] = list(unexpected)
            
            # Log reconciliation results
            logger.info("\nReconciliation Results:")
            logger.info(f"  ✓ Matched orders: {len(matched)}")
            
            if missing:
                logger.warning(f"  ⚠ Missing orders: {len(missing)}")
                for order_id in missing:
                    logger.warning(f"    - {order_id} (was expected, not found on exchange)")
            
            if unexpected:
                logger.warning(f"  ⚠ Unexpected orders: {len(unexpected)}")
                for order_id in unexpected:
                    logger.warning(f"    - {order_id} (found on exchange, not in saved state)")
            
            # Handle mismatches
            if unexpected:
                logger.info("\nHandling unexpected orders...")
                cancelled = self._cancel_unexpected_orders(
                    connector, pair, list(unexpected)
                )
                reconciliation['actions_taken'].append({
                    'action': 'cancel_unexpected',
                    'order_ids': cancelled,
                    'count': len(cancelled)
                })
            
            if missing:
                logger.info("\nMissing orders detected - grid may need rebuilding")
                reconciliation['actions_taken'].append({
                    'action': 'mark_for_rebuild',
                    'reason': f'{len(missing)} orders missing',
                    'missing_order_ids': list(missing)
                })
            
            logger.info("\n" + "=" * 60)
            logger.info("RECONCILIATION COMPLETE")
            logger.info("=" * 60)
            
            return reconciliation
        
        except Exception as e:
            logger.error(f"Error during reconciliation: {e}")
            reconciliation['error'] = str(e)
            return reconciliation
    
    def _cancel_unexpected_orders(
        self,
        connector: GateIOConnector,
        pair: str,
        order_ids: List[str]
    ) -> List[str]:
        """
        Cancel unexpected orders found during reconciliation.
        
        Args:
            connector: Gate.io connector
            pair: Trading pair
            order_ids: List of order IDs to cancel
            
        Returns:
            List of successfully cancelled order IDs
        """
        cancelled = []
        
        for order_id in order_ids:
            try:
                def _cancel_order():
                    return connector._request(
                        "DELETE",
                        f"/spot/orders/{order_id}",
                        params={"currency_pair": pair},
                        auth=True
                    )
                
                connector._retry_on_error(_cancel_order)
                cancelled.append(order_id)
                logger.info(f"  ✓ Cancelled unexpected order: {order_id}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to cancel order {order_id}: {e}")
        
        return cancelled
    
    def needs_grid_rebuild(self, reconciliation: Dict[str, Any]) -> bool:
        """
        Determine if grid needs to be rebuilt based on reconciliation.
        
        Grid rebuild is needed if:
        - Too many orders are missing
        - Significant mismatch between expected and actual
        
        Args:
            reconciliation: Reconciliation results dictionary
            
        Returns:
            bool: True if grid should be rebuilt
            
        Example:
            >>> if manager.needs_grid_rebuild(state['reconciliation']):
            ...     print("Rebuilding grid...")
            ...     # Rebuild logic here
        """
        if not reconciliation:
            return False
        
        expected_count = len(reconciliation.get('expected_orders', []))
        matched_count = len(reconciliation.get('matched_orders', []))
        missing_count = len(reconciliation.get('missing_orders', []))
        
        # Rebuild if more than 50% of orders are missing
        if expected_count > 0:
            missing_pct = (missing_count / expected_count) * 100
            
            if missing_pct > 50:
                logger.warning(
                    f"Grid rebuild needed: {missing_pct:.1f}% of orders missing "
                    f"({missing_count}/{expected_count})"
                )
                return True
        
        # Rebuild if no orders matched but we expected some
        if expected_count > 0 and matched_count == 0:
            logger.warning("Grid rebuild needed: no orders matched")
            return True
        
        logger.info("Grid rebuild not needed")
        return False
    
    def _backup_state(self) -> bool:
        """
        Create a backup of the current state file.
        
        Returns:
            bool: True if backup successful
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(
                self.backup_dir,
                f"bot_state_backup_{timestamp}.json"
            )
            
            # Copy current state file to backup
            with open(self.state_file, 'r') as src:
                with open(backup_file, 'w') as dst:
                    dst.write(src.read())
            
            logger.debug(f"State backed up to: {backup_file}")
            
            # Clean old backups (keep last 10)
            self._cleanup_old_backups(keep=10)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to backup state: {e}")
            return False
    
    def _cleanup_old_backups(self, keep: int = 10) -> None:
        """
        Remove old backup files, keeping only the most recent.
        
        Args:
            keep: Number of recent backups to keep (default: 10)
        """
        try:
            backup_files = sorted(
                Path(self.backup_dir).glob('bot_state_backup_*.json'),
                key=os.path.getmtime,
                reverse=True
            )
            
            # Remove old backups
            for old_backup in backup_files[keep:]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
        
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def get_recovery_summary(self, state: Dict[str, Any]) -> str:
        """
        Generate a human-readable recovery summary.
        
        Args:
            state: Loaded state with reconciliation results
            
        Returns:
            str: Formatted recovery summary
            
        Example:
            >>> state = manager.load_state(client, 'SOL_USDT')
            >>> print(manager.get_recovery_summary(state))
        """
        if not state:
            return "No state to summarize"
        
        lines = []
        lines.append("=" * 60)
        lines.append("RECOVERY SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        
        # Basic state info
        lines.append(f"State Timestamp: {state.get('timestamp', 'Unknown')}")
        lines.append(f"Last Equity: ${state.get('last_equity', 0):.2f}")
        lines.append("")
        
        # Balances
        balances = state.get('balances', {})
        lines.append("Balances:")
        for currency, amount in balances.items():
            lines.append(f"  {currency}: {amount:.4f}")
        lines.append("")
        
        # Grid parameters
        grid_params = state.get('grid_parameters', {})
        lines.append("Grid Parameters:")
        lines.append(f"  Pair: {grid_params.get('pair', 'N/A')}")
        lines.append(f"  Buy Levels: {len(grid_params.get('buy_levels', []))}")
        lines.append(f"  Sell Levels: {len(grid_params.get('sell_levels', []))}")
        lines.append("")
        
        # Reconciliation results
        if 'reconciliation' in state:
            recon = state['reconciliation']
            lines.append("Reconciliation Results:")
            lines.append(f"  Expected Orders: {len(recon.get('expected_orders', []))}")
            lines.append(f"  Actual Orders: {len(recon.get('actual_orders', []))}")
            lines.append(f"  Matched: {len(recon.get('matched_orders', []))}")
            lines.append(f"  Missing: {len(recon.get('missing_orders', []))}")
            lines.append(f"  Unexpected: {len(recon.get('unexpected_orders', []))}")
            lines.append("")
            
            # Actions taken
            actions = recon.get('actions_taken', [])
            if actions:
                lines.append("Actions Taken:")
                for action in actions:
                    lines.append(f"  • {action.get('action', 'Unknown')}")
                lines.append("")
            
            # Rebuild recommendation
            if self.needs_grid_rebuild(recon):
                lines.append("⚠️  RECOMMENDATION: Rebuild grid")
            else:
                lines.append("✓ Grid state is consistent")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def clear_state(self) -> bool:
        """
        Clear the current state file.
        
        This should be used when starting fresh or after a complete reset.
        
        Returns:
            bool: True if clear successful
            
        Example:
            >>> manager.clear_state()  # Start fresh
        """
        try:
            if os.path.exists(self.state_file):
                # Backup before clearing
                self._backup_state()
                os.remove(self.state_file)
                logger.info("State file cleared")
            
            self.current_state = None
            return True
        
        except Exception as e:
            logger.error(f"Failed to clear state: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage demonstrating state save/load and recovery.
    """
    
    print("\n" + "=" * 70)
    print("STATE SYNCHRONIZATION - Example Usage")
    print("=" * 70)
    print()
    
    # Initialize state manager
    manager = StateManager(state_file='test_bot_state.json')
    
    # Example 1: Save state
    print("-" * 70)
    print("Example 1: Saving Bot State")
    print("-" * 70)
    print()
    
    # Sample data
    balances = {
        'USDT': 9500.50,
        'SOL': 1.25
    }
    
    active_orders = [
        {
            'order_id': '12345678',
            'side': 'buy',
            'price': 98.0,
            'amount': 0.1,
            'pair': 'SOL_USDT'
        },
        {
            'order_id': '12345679',
            'side': 'buy',
            'price': 96.0,
            'amount': 0.1,
            'pair': 'SOL_USDT'
        },
        {
            'order_id': '12345680',
            'side': 'sell',
            'price': 102.0,
            'amount': 0.1,
            'pair': 'SOL_USDT'
        },
        {
            'order_id': '12345681',
            'side': 'sell',
            'price': 104.0,
            'amount': 0.1,
            'pair': 'SOL_USDT'
        }
    ]
    
    grid_parameters = {
        'pair': 'SOL_USDT',
        'buy_levels': [98.0, 96.0, 94.0],
        'sell_levels': [102.0, 104.0, 106.0],
        'base_amount': 0.1,
        'grid_range_pct': 0.10,
        'total_levels': 6,
        'trend_bias': 'neutral'
    }
    
    last_equity = 10125.75
    
    metadata = {
        'bot_version': '1.0.0',
        'trading_mode': 'testnet',
        'strategy': 'grid_scalping'
    }
    
    success = manager.save_state(
        balances=balances,
        active_orders=active_orders,
        grid_parameters=grid_parameters,
        last_equity=last_equity,
        metadata=metadata
    )
    
    if success:
        print("✓ State saved successfully")
    print()
    
    # Example 2: Load state (without reconciliation)
    print("-" * 70)
    print("Example 2: Loading Bot State (No Reconciliation)")
    print("-" * 70)
    print()
    
    loaded_state = manager.load_state(reconcile=False)
    
    if loaded_state:
        print("✓ State loaded successfully")
        print(f"  Timestamp: {loaded_state['timestamp']}")
        print(f"  Equity: ${loaded_state['last_equity']:.2f}")
        print(f"  Orders: {len(loaded_state['active_orders'])}")
        print(f"  Pair: {loaded_state['grid_parameters']['pair']}")
    print()
    
    # Example 3: Recovery summary
    print("-" * 70)
    print("Example 3: Recovery Summary")
    print("-" * 70)
    print()
    
    print(manager.get_recovery_summary(loaded_state))
    
    # Example 4: State reconciliation (with mock connector - commented out)
    print("-" * 70)
    print("Example 4: State Reconciliation (REQUIRES LIVE CONNECTION)")
    print("-" * 70)
    print()
    
    print("⚠️  To test reconciliation, uncomment the code below and provide")
    print("valid API credentials in .env file")
    print()
    
    """
    # Uncomment to test with real connection
    from connectors.gate_io import initialize_client
    
    client = initialize_client('testnet')
    state_with_recon = manager.load_state(
        connector=client,
        pair='SOL_USDT',
        reconcile=True
    )
    
    if state_with_recon and 'reconciliation' in state_with_recon:
        print("Reconciliation completed:")
        recon = state_with_recon['reconciliation']
        print(f"  Matched: {len(recon['matched_orders'])}")
        print(f"  Missing: {len(recon['missing_orders'])}")
        print(f"  Unexpected: {len(recon['unexpected_orders'])}")
        
        if manager.needs_grid_rebuild(recon):
            print("\n⚠️  Grid rebuild recommended")
    """
    
    # Cleanup
    print("-" * 70)
    print("Cleanup")
    print("-" * 70)
    print()
    
    if os.path.exists('test_bot_state.json'):
        os.remove('test_bot_state.json')
        print("✓ Test state file removed")
    
    print()
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
