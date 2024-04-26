
from utils.run_servers import run_servers

if __name__ == "__main__":
    # Output the URL for the user to visit
    print("Backend running at http://127.0.0.1:5000")
    print("Frontend running at http://127.0.0.1:4200")

    # Run both backend API and frontend servers
    run_servers()