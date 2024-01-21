import subprocess

def run_servers():
    """Run both backend API and frontend servers."""
    try:
        # Wait for both processes to finish
        subprocess.run("bash start_servers.sh", shell=True)
    except KeyboardInterrupt:
        # If the user interrupts (Ctrl+C), terminate both processes
        raise KeyboardInterrupt("Program terminated by user!")