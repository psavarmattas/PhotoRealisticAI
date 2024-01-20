import time

def cooldown(duration_minutes):
    """
    Pause training for the specified duration.

    Args:
        duration_minutes (int): Duration of the cooldown period in minutes.

    Returns:
        None
    """
    duration_seconds = duration_minutes * 60
    remaining_time = duration_seconds

    print("====================================================================================================")
    while remaining_time > 0:
        minutes, seconds = divmod(remaining_time, 60)
        print(f"Cooldown Mode:: {int(minutes):02}:{int(seconds):02} remaining...", end="\r")
        time.sleep(1)
        remaining_time -= 1

    print("Cooldown completed. Resuming training...")
    print("====================================================================================================")