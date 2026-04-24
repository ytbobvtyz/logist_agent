#!/usr/bin/env python3
"""
Test if the server starts correctly.
"""

import sys
import time
import threading
import requests
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_server_start():
    """Test if server can start and respond."""
    from app.main import LogsitApp
    
    print("Testing server startup...")
    
    app = LogsitApp()
    
    try:
        print("1. Initializing application...")
        app.initialize()
        print("  ✅ Initialization complete")
        
        print("2. Creating UI...")
        app.create_ui()
        print("  ✅ UI created")
        
        # Start server in background thread
        print("3. Starting server in background thread...")
        
        def run_in_thread():
            try:
                app.run()
            except Exception as e:
                print(f"  ❌ Server error in thread: {e}")
        
        server_thread = threading.Thread(target=run_in_thread, daemon=True)
        server_thread.start()
        
        # Give server time to start
        print("4. Waiting for server to start...")
        time.sleep(3)
        
        # Try to connect to server
        print("5. Testing server connection...")
        try:
            response = requests.get("http://localhost:7860", timeout=5)
            if response.status_code == 200:
                print(f"  ✅ Server is running! Status: {response.status_code}")
                return True
            else:
                print(f"  ⚠️ Server responded with status: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("  ❌ Could not connect to server")
            return False
        except Exception as e:
            print(f"  ❌ Connection error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Logsit Agent - Server Health Check")
    print("=" * 60)
    
    success = test_server_start()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SERVER TEST PASSED")
        print("\nThe application should be available at:")
        print("  http://localhost:7860")
        print("\nPress Ctrl+C to stop the server.")
        
        # Keep running to allow user to test
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
    else:
        print("❌ SERVER TEST FAILED")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())