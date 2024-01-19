import subprocess
import sys

def check_dependencies():

    """
    Check dependencies listed in the requirements.txt file.

    This function reads the 'requirements.txt' file, extracts the dependencies,
    and checks if each dependency is installed. If any dependencies are missing,
    it prints a list of missing dependencies and exits the program with an error code.

    Raises:
    FileNotFoundError: If 'requirements.txt' is not found.
    SystemExit: If any dependencies are missing, the function exits with status code 1.

    Returns:
    None
    """

    print("====================================================================================================")
    print("Checking dependencies...")
    print("====================================================================================================")
    try:
        with open('requirements.txt', 'r') as file:
            dependencies = [line.strip() for line in file if line.strip()]

        missing_dependencies = []
        for dependency in dependencies:
            try:
                # Check if the library is installed
                subprocess.check_output(['pip', 'show', dependency])
            except subprocess.CalledProcessError:
                missing_dependencies.append(dependency)

        if missing_dependencies:
            print("Missing dependencies:")
            for missing_dependency in missing_dependencies:
                print(f"- {missing_dependency}")

            print("Please install the missing dependencies before proceeding.")
            sys.exit(1)

    except FileNotFoundError:
        print("Error: requirements.txt not found.")
        sys.exit(1)