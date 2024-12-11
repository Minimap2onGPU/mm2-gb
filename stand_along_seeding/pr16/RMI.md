## `rmi.h` break down
### Class Template: `RMI<rmi_key_t>`
- **Generic Type**: The template `rmi_key_t` is used to define the type of the keys that RMI will handle, allowing flexibility in the data type (e.g., `double`, `uint64_t`).

### Constructor: `RMI(char *prefix)`
- **Purpose**: Initializes an RMI instance by loading sorted array data and RMI parameters from files.
- **Operations**:
  - Calls `load_sorted_array` to load keys from a binary file.
  - Calls `load_rmi` to load model parameters for the root and leaf nodes of the RMI.

### Destructor: `~RMI()`
- **Purpose**: Cleans up resources by freeing allocated memory for the sorted array and model parameters.

### Method: `get_random_keys(int64_t nq, rmi_key_t *key_array, int64_t *orig_pos_array)`
- **Purpose**: Generates a random set of query keys from the sorted keys to simulate query operations, mainly used for testing and benchmarking.
- **Operations**:
  - Randomly selects `nq` keys from the sorted array.

### Method: `get_element(int64_t index)`
- **Purpose**: Retrieves an element from the sorted array at a specified index, ensuring the index is within bounds.
- **Operations**:
  - Checks if the index is valid and returns the corresponding element from the sorted array.

### Method: `load_sorted_array(char *prefix)`
- **Purpose**: Loads the sorted array from a file which contains the keys that have been pre-sorted.
- **Operations**:
  - Opens the file containing sorted keys and reads them into the `sorted_array`.

### Method: `load_rmi(char *prefix)`
- **Purpose**: Loads the RMI model parameters from a file.
- **Operations**:
  - Reads root and leaf model parameters that are used to predict the positions of keys within the sorted array.

### Method: `get_guess(rmi_key_t key, int64_t *err)`
- **Purpose**: Uses the RMI to estimate the position of a key in the sorted array, returning the guess and the error margin.
- **Operations**:
  - Calculates a predicted position using a two-level model (root and leaf), with error handling to capture prediction inaccuracies.

### Method: `last_mile_search(rmi_key_t key, int64_t guess, size_t err)`
- **Purpose**: Performs a localized search around the predicted position to find the exact position of the key.
- **Operations**:
  - Uses a binary search within the error bounds specified by `err` to pinpoint the exact position or the nearest element.

### Method: `lookup(rmi_key_t key)`
- **Purpose**: Public interface to perform a complete lookup for a key, integrating guess computation and last mile search.
- **Operations**:
  - Orchestrates a full lookup process, starting from RMI prediction to last mile search, returning the position of the key or an error if not found.

### Method: `lookup_batched(rmi_key_t *key_array, int64_t num_queries, int64_t *pos_array)`
- **Purpose**: Handles multiple queries at once, useful for processing bulk queries efficiently.
- **Operations**:
  - Processes each key in the batch through the RMI and updates their respective positions in `pos_array`.

### Method: `print_stats()`
- **Purpose**: Outputs statistical data about the RMI operations, primarily used for debugging and performance analysis.
- **Operations**:
  - Prints various statistics such as average error, maximum error, and frequency of errors.

## RMI
### Key-Sorted List of Key-Value Pairs
1. **Keys**: The keys are the actual minimizers, which are unique subsequences identified within the genome data. These keys are sorted in some order, typically lexicographical, to facilitate quick lookup.

2. **Values**: The values associated with each key in this sorted list are structured to provide quick access to their locations within the genome. Specifically, each value is composed of:
   - **Starting Index in the Position List**: This index points to where the list of positions for this specific minimizer begins in a separate position list.
   - **Count of the Positions**: This count tells how many times the minimizer appears in the dataset, which helps in understanding how many entries to read from the position list starting from the provided index.

### Position List
- This is a concatenated list of all positions of all minimizers. Each position indicates where a particular minimizer can be found in the reference genome sequence.

### Example Explained:
- **Minimizer Entry**: `mm5 → [8,3]`
   - This indicates that the minimizer `mm5` is found 3 times in the reference sequence.
   - The numbers `8, 3` mean the positions of `mm5` start from index 8 in the position list, and there are 3 positions to read from this starting point.
   - If the positions are located at indexes 5, 21, and 57 in the genome sequence, then starting at index 8 of the position list, you would find these three numbers listed consecutively.