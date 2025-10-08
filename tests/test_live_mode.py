import pytest
import subprocess
import os
import time
import tempfile
import shutil

@pytest.fixture
def test_script():
    """Create a temporary test script file"""
    script_dir = os.path.dirname(__file__)
    script_content = os.path.join(script_dir, 'live_test_tagger.py')
    return script_content

@pytest.fixture
def test_files():
    """Create temporary test files for the test"""
    test_dir = tempfile.mkdtemp()
    test_files = []
    
    for i in range(3):
        test_file = os.path.join(test_dir, f"test_image_{i}.jpg")
        with open(test_file, 'w') as f:
            f.write(f"dummy image content {i}")
        test_files.append(test_file)
    
    yield test_files
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)

@pytest.fixture(scope='session')
def output_dir():
    """Create and clean output directory"""
    output_path = tempfile.mkdtemp()
    yield output_path
    # Cleanup
    shutil.rmtree(output_path, ignore_errors=True)

def create_test_tag_fn(output_dir):
    """Create a test tag function that outputs batch size to a file"""
    def test_tag_fn(file_paths):
        batch_size = len(file_paths)
        output_file = os.path.join(output_dir, "batch_output.txt")
        with open(output_file, 'w') as f:
            f.write(str(batch_size))
    return test_tag_fn

def test_live_mode_slow_files(test_files, test_script):
    """Test live mode with files sent slowly (should get batch size 1)"""
    
    # Start the live mode process
    proc = subprocess.Popen(
        ['python', test_script],
        stdin=subprocess.PIPE,
        text=True,
    )
    
    try:
        # Send one file, wait for processing, then close
        assert proc.stdin is not None
        for fpath in test_files:
            proc.stdin.write(f"{fpath}\n")
            proc.stdin.flush()
            
            # Wait long enough for processing (longer than batch_timeout of 0.2s)
            time.sleep(0.5)
        
        # check there is an output file for each test file with a value of 1
        for fpath in test_files:
            output_dir = os.path.dirname(fpath)
            output_file = os.path.join(output_dir, f"{os.path.basename(fpath)}.txt")
            assert os.path.exists(output_file), f"Output file {output_file} was not created"
            with open(output_file, 'r') as f:
                content = f.read().strip()
            assert content == "1", f"Expected batch size 1 for {fpath}, got {content}"
        
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail("Process timed out")
    finally:
        proc.kill()
        proc.wait()

def test_live_mode_batch_files(test_files, output_dir, test_script):
    """Test live mode with files sent quickly (should get batch size 3)"""
    
    # Start the live mode process
    proc = subprocess.Popen(
        ['python', test_script],
        stdin=subprocess.PIPE,
        text=True,
    )
    
    try:
        # Send all files quickly
        assert proc.stdin is not None
        input_data = '\n'.join(test_files) + '\n'
        proc.stdin.write(input_data)
        proc.stdin.flush()

        time.sleep(0.5)

        for fpath in test_files:
            output_dir = os.path.dirname(fpath)
            output_file = os.path.join(output_dir, f"{os.path.basename(fpath)}.txt")
            assert os.path.exists(output_file), f"Output file {output_file} was not created"
            with open(output_file, 'r') as f:
                content = f.read().strip()
            assert content == "3", f"Expected batch size 3 for {fpath}, got {content}"
        
    finally:
        proc.kill()
        proc.wait()

def test_live_mode_batch_files_with_delay(test_files, output_dir, test_script):
    """Test live mode with startup delay but files sent quickly (should get batch size 3)"""
    
    # Start the live mode process with a 1 second delay
    proc = subprocess.Popen(
        ['python', test_script, '--delay', '1'],
        stdin=subprocess.PIPE,
        text=True,
    )
    
    try:
        # Send all files quickly right away (even though tagger has startup delay)
        assert proc.stdin is not None
        for fpath in test_files:
            proc.stdin.write(f"{fpath}\n")
            proc.stdin.flush()
        
        # Wait a bit longer for the delayed startup and processing
        time.sleep(2)

        for fpath in test_files:
            output_dir = os.path.dirname(fpath)
            output_file = os.path.join(output_dir, f"{os.path.basename(fpath)}.txt")
            assert os.path.exists(output_file), f"Output file {output_file} was not created"
            with open(output_file, 'r') as f:
                content = f.read().strip()
            assert content == "3", f"Expected batch size 3 for {fpath}, got {content}"
        
    finally:
        proc.kill()
        proc.wait()