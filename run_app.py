#!/usr/bin/env python3
"""
Simple script to run Logsit Agent app.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """Check that all required modules can be imported."""
    print("Checking imports...")
    
    modules_to_check = [
        ("utils.config", "settings"),
        ("utils.async_helpers", "start_background_loop"),
        ("core.agent", "LogsitAgent"),
        ("app.components.sidebar", "SidebarComponent"),
        ("app.components.chat", "ChatComponent"),
        ("app.handlers.message_handler", "MessageHandler"),
    ]
    
    all_ok = True
    for module_name, item_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[item_name])
            getattr(module, item_name)
            print(f"  ✅ {module_name}.{item_name}")
        except Exception as e:
            print(f"  ❌ {module_name}.{item_name}: {e}")
            all_ok = False
    
    return all_ok

def main():
    """Main entry point."""
    print("=" * 60)
    print("Logsit Agent - Application Launcher")
    print("=" * 60)
    
    # Check imports first
    if not check_imports():
        print("\n❌ Some imports failed. Please install package first:")
        print("   pip install -e .")
        return 1
    
    print("\nStarting application...")
    
    try:
        from app.main import main as app_main
        app_main()
        return 0
    except KeyboardInterrupt:
        print("\n\n✅ Application stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())