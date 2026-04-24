#!/usr/bin/env python3
"""
Test script to verify fixes.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test configuration."""
    print("Testing configuration...")
    from utils.config import settings
    
    print(f"OpenRouter API key: {'Set' if settings.openrouter_api_key else 'Not set'}")
    print(f"Use OpenRouter: {settings.use_openrouter}")
    
    # Test MCP servers parsing
    test_servers = "test1:python server1.py,test2:python server2.py"
    settings.mcp_servers = test_servers
    servers = settings.get_mcp_servers_dict()
    print(f"MCP servers parsed: {len(servers)} servers")
    
    return True

def test_simple_app():
    """Test simple app initialization without UI."""
    print("\nTesting app initialization...")
    
    try:
        from utils.async_helpers import start_background_loop, stop_background_loop
        from core.agent import LogsitAgent
        
        print("1. Starting background loop...")
        start_background_loop()
        
        print("2. Creating agent...")
        agent = LogsitAgent()
        
        print("3. Testing agent methods...")
        # Test create conversation
        conv_id = agent.conversation_manager.create_conversation()
        print(f"   Created conversation: {conv_id}")
        
        print("✅ App initialization test passed")
        return True
    except Exception as e:
        print(f"❌ App initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        try:
            stop_background_loop()
        except:
            pass

def test_openai_api():
    """Test OpenAI/OpenRouter API connection."""
    print("\nTesting API connection...")
    
    try:
        import openai
        from utils.config import settings
        
        # Use OpenRouter if available
        if settings.openrouter_api_key:
            client = openai.OpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Simple test call
            try:
                response = client.chat.completions.create(
                    model="google/gemma-3-27b-it:free",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                print(f"✅ API connection successful (OpenRouter)")
                return True
            except Exception as e:
                print(f"❌ OpenRouter API error: {e}")
                
                # Try OpenAI if OpenRouter fails
                if settings.openai_api_key:
                    client = openai.OpenAI(api_key=settings.openai_api_key)
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": "Hello"}],
                            max_tokens=10
                        )
                        print(f"✅ API connection successful (OpenAI)")
                        return True
                    except Exception as e2:
                        print(f"❌ OpenAI API error: {e2}")
                        return False
                else:
                    return False
        else:
            print("⚠️ No API keys configured")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Logsit Agent - Fix Verification Test")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_config),
        ("API Connection", test_openai_api),
        ("App Initialization", test_simple_app),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Application should work correctly.")
        print("\nTo run the application:")
        print("  cd /home/ytbob/projects/logsit_agent")
        print("  source venv/bin/activate")
        print("  python app/main.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed.")
        print("\nCommon issues:")
        print("1. API keys not configured in .env file")
        print("2. OpenAI/OpenRouter API region restrictions")
        print("3. Missing dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main())