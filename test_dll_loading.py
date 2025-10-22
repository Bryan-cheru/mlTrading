"""
Simple Rithmic DLL Test - Bypass .NET security restrictions
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))

def test_dll_loading():
    """Test different methods to load the Rithmic DLL"""
    
    print("🧪 TESTING DLL LOADING METHODS")
    print("=" * 50)
    
    dll_path = os.path.join(project_root, "13.6.0.0", "win10", "lib_472", "rapiplus.dll")
    local_dll = os.path.join(project_root, "rithmic-dll", "rapiplus.dll")
    
    print(f"Original DLL: {dll_path}")
    print(f"Local DLL: {local_dll}")
    print(f"DLL exists: {os.path.exists(dll_path)}")
    print(f"DLL size: {os.path.getsize(dll_path):,} bytes")
    
    # Method 1: Direct pythonnet
    print("\n1️⃣ Testing pythonnet...")
    try:
        import clr
        
        # Try loading by filename only (from current directory)
        current_dll = os.path.join(project_root, "rapiplus.dll")
        if not os.path.exists(current_dll):
            import shutil
            shutil.copy2(dll_path, current_dll)
        
        clr.AddReference("rapiplus")
        print("   ✅ DLL loaded via current directory reference")
        
        # Test basic import
        try:
            import System
            print("   ✅ System namespace imported")
        except:
            print("   ❌ System namespace import failed")
            
    except Exception as e:
        print(f"   ❌ pythonnet failed: {e}")
    
    # Method 2: ctypes
    print("\n2️⃣ Testing ctypes...")
    try:
        import ctypes
        dll_handle = ctypes.CDLL(dll_path)
        print("   ✅ DLL loaded via ctypes")
        print(f"   Handle: {dll_handle}")
    except Exception as e:
        print(f"   ❌ ctypes failed: {e}")
    
    # Method 3: Alternative .NET loading
    print("\n3️⃣ Testing alternative CLR loading...")
    try:
        import clr
        import System
        
        # Use Assembly.UnsafeLoadFrom
        dll_handle = System.Reflection.Assembly.UnsafeLoadFrom(dll_path)
        print(f"   ✅ DLL loaded via UnsafeLoadFrom: {dll_handle}")
        
    except Exception as e:
        print(f"   ❌ UnsafeLoadFrom failed: {e}")
    
    print("\n✨ Test completed")

if __name__ == "__main__":
    test_dll_loading()