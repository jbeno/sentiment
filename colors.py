# colors.py

# Format some colors
black = "\033[30m"
red = "\033[31m"
green = "\033[32m"
yellow = "\033[33m"
blue = "\033[34m"
magenta = "\033[35m"
cyan = "\033[36m"
white = "\033[37m"
reset = "\033[0m"
bold = "\033[1m"
underline = "\033[4m"
reverse = "\033[7m"
orange = "\033[38;5;208m"
bright_green = "\033[38;5;82m"
bright_yellow = "\033[93m"
bright_red = "\033[91m"
bright_white = "\033[97m"
bright_magenta = "\033[95m"
bright_cyan = "\033[96m"
bright_blue = "\033[94m"
bright_purple = "\033[38;5;129m"
light_blue = "\033[38;5;123m"
sky_blue = "\033[38;5;117m"
deep_blue = "\033[38;5;21m"
steel_blue = "\033[38;5;67m"
electric_blue = "\033[38;5;81m"
purple = "\033[38;5;93m"


def get_nic_color(no_improvement_count, n_iter_no_change):
    """
    Determine the color for the no improvement count based on its value 
    relative to n_iter_no_change.

    Args:
    no_improvement_count (int): The current count of iterations with no improvement.
    n_iter_no_change (int): The maximum number of iterations allowed with no improvement.

    Returns:
    str: The ANSI color code for the no improvement count.
    """
    if no_improvement_count == 0:
        return bright_green
    elif no_improvement_count < n_iter_no_change / 3:
        return bright_yellow
    elif no_improvement_count < 2 * n_iter_no_change / 3:
        return orange
    else:
        return bright_red

def get_score_colors(current_score, best_score):
    """
    Determine the colors for displaying the current score and best score based on their values.

    This function assigns colors to the current score and best score for visual representation.
    The color scheme is as follows:
    - Bright red for scores below 0.50
    - Orange for scores between 0.50 and 0.59
    - Bright yellow for scores between 0.60 and 0.69
    - Bright green for scores 0.70 and above

    The function compares the current score with the best score (if available) and assigns
    colors accordingly:
    - If there's no best score yet, it colors the current score based on its value and uses
      bright white for the best score.
    - If the current score is greater than or equal to the best score, it colors the current
      score based on its value and uses bright white for the best score.
    - If the best score is higher, it uses bright white for the current score and colors the
      best score based on its value.

    Args:
    current_score (float): The current score to be evaluated.
    best_score (float or None): The best score achieved so far, or None if no best score exists.

    Returns:
    tuple: A tuple containing two color codes (strings):
           (color for current score, color for best score)

    Usage:
    current_score_color, best_score_color = get_score_colors(current_score, self.best_score)
    """
    def get_color(score):
        if score < .50:
            return bright_red
        elif score < .60:
            return orange
        elif score < .70:
            return bright_yellow
        else:
            return bright_green

    if best_score is None:
        return get_color(current_score), bright_white

    if current_score >= best_score:
        return get_color(current_score), bright_white
    else:
        return bright_white, get_color(best_score)

    
def get_shape_color(batch_size, shape):
    """
    Determine the color for displaying the shape based on its first dimension 
    compared to the batch size.

    Args:
    batch_size (int): The expected batch size.
    shape (tuple): The shape of the tensor or array.

    Returns:
    str: ANSI color code (bright_white or bright_yellow).

    Usage:
    shape_color = get_shape_color(batch_size, tensor.shape)
    """
    if shape[0] == batch_size:
        return bright_white
    else:
        return bright_yellow

def get_nic_color(no_improvement_count, n_iter_no_change):
    """
    Determine the color for the no improvement count based on its value 
    relative to n_iter_no_change.

    Args:
    no_improvement_count (int): The current count of iterations with no improvement.
    n_iter_no_change (int): The maximum number of iterations allowed with no improvement.

    Returns:
    str: The ANSI color code for the no improvement count.
    """
    if no_improvement_count == 0:
        return bright_green
    elif no_improvement_count < n_iter_no_change / 3:
        return bright_yellow
    elif no_improvement_count < 2 * n_iter_no_change / 3:
        return orange
    else:
        return bright_red
    
    
def get_shape_color(batch_size, shape):
    """
    Determine the color for displaying the shape based on its first dimension 
    compared to the batch size.

    Args:
    batch_size (int): The expected batch size.
    shape (tuple): The shape of the tensor or array.

    Returns:
    str: ANSI color code (bright_white or bright_yellow).

    Usage:
    shape_color = get_shape_color(batch_size, tensor.shape)
    """
    if shape[0] == batch_size:
        return bright_white
    else:
        return bright_yellow

def get_mem_color(memory, max_memory):
    """
    Get color for memory usage based on its proportion of max_memory.
    
    Args:
    memory (float or list): Current memory usage(s).
    max_memory (float): Maximum memory across all GPUs.

    Returns:
    str or list: Color code(s) for the given memory usage(s).
    """
    def _get_single_color(mem):
        proportion = mem / max_memory
        if proportion < 0.5:
            return bright_green
        elif proportion < 0.75:
            return bright_yellow
        elif proportion < 0.9:
            return orange
        else:
            return bright_red

    if isinstance(memory, (list, tuple)):
        return [_get_single_color(mem) for mem in memory]
    else:
        return _get_single_color(memory)
    
def get_loss_colors(current_loss, best_error, no_improvement_count, n_iter_no_change):
    """
    Determine colors for current loss and best error based on their relationship and improvement count.

    Args:
    current_loss (float): The current loss value to be evaluated.
    best_error (float): The best (lowest) error achieved so far.
    no_improvement_count (int): The number of iterations without improvement.
    n_iter_no_change (int): The maximum number of iterations allowed without improvement.

    Returns:
    tuple: (color for current loss, color for best error)
    """
    if current_loss <= best_error:
        current_loss_color = bright_green
        best_error_color = bright_white
    else:
        current_loss_color = get_nic_color(no_improvement_count, n_iter_no_change)
        best_error_color = bright_green

    return current_loss_color, best_error_color