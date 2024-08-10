import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import time
from multiprocessing import Queue, Pipe, set_start_method

from utils import setup_environment, prepare_device, cleanup_and_exit, fix_random_seeds

def run(rank, world_size, device_type, backend, debug, input_queue, response_pipe):
    device = prepare_device(rank, device_type)
    setup_environment(rank, world_size, backend, device, debug)
    fix_random_seeds(42)

    num_epochs = 5
    user_input = 'c'

    # Create a global tensor for synchronization
    global_tensor = torch.zeros(2, dtype=torch.long, device=device)

    for epoch in range(num_epochs):
        # Simulate epoch work
        time.sleep(1)  # Simulate some work being done

        print(f"Rank {rank} completed epoch {epoch + 1}")

        # Synchronize all processes after each epoch
        dist.barrier()

        if rank == 0:
            print(f"Epoch {epoch + 1} completed. Waiting for input...")
            sys.stdout.flush()
            input_queue.put(rank)
            user_input = response_pipe.recv()
            print(f"Rank {rank} received input: {user_input}")
            
            # Set the global tensor
            global_tensor[0] = 1  # Signal that input is ready
            global_tensor[1] = ord(user_input[0])  # Store the first character of input
        
        # Broadcast the global tensor to all processes
        dist.broadcast(global_tensor, src=0)
        
        # All processes wait until the input is ready
        while global_tensor[0].item() == 0:
            dist.broadcast(global_tensor, src=0)
            time.sleep(0.1)
        
        # Process the input
        user_input = chr(global_tensor[1].item())
        print(f"Rank {rank} processed input: {user_input}")

        if user_input == 'q':
            print(f"Rank {rank} received quit command. Exiting.")
            break

        # Reset the global tensor for the next iteration
        if rank == 0:
            global_tensor[0] = 0
        dist.broadcast(global_tensor, src=0)

        # Ensure all ranks have processed the input before continuing
        dist.barrier()

    cleanup_and_exit(rank, debug)

def main(rank, world_size, device_type, backend, debug, input_queue, pipes):
    try:
        response_pipe = pipes[rank][1]  # Get the specific pipe for this rank
        run(rank, world_size, device_type, backend, debug, input_queue, response_pipe)
    except Exception as e:
        print(f"Rank {rank} encountered an error: {e}")
        cleanup_and_exit(rank, debug)

if __name__ == "__main__":
    set_start_method('spawn', force=True)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    world_size = torch.cuda.device_count() if device_type == "cuda" else 1
    backend = "nccl" if device_type == "cuda" else "gloo"
    debug = True

    print(f"Device: {device_type}, World Size: {world_size}, Backend: {backend}")

    input_queue = Queue()
    pipes = [Pipe() for _ in range(world_size)]

    try:
        processes = mp.spawn(main,
                             args=(world_size, device_type, backend, debug, input_queue, pipes),
                             nprocs=world_size,
                             join=False)

        while True:
            if not input_queue.empty():
                rank = input_queue.get()
                user_input = input("Enter a command ('c' to continue, 'q' to quit): ").strip().lower()
                pipes[rank][0].send(user_input)
                if user_input == 'q':
                    break
            time.sleep(0.1)  # Small sleep to prevent busy waiting

        processes.join()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Terminating all processes...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cleanup_and_exit(0, debug)

    print("All processes finished.")