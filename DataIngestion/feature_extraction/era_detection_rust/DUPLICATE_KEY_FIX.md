# Era Detection Duplicate Key Fix

## Problem
The era detection was failing with duplicate key constraint violations:
```
ERROR: duplicate key value violates unique constraint "72412_240_era_labels_level_b_pkey"
DETAIL: Key (signal_name, level, stage, era_id, start_time)=(vent_pos_2_percent, B, BOCPD, 1, 2013-12-01 00:00:00+00) already exists.
```

## Root Cause
1. Multiple signals were being processed in parallel (`signal_columns.par_iter()`)
2. Each signal processing thread was:
   - First calling `delete_era_labels_for_signal()` to delete old data
   - Then calling `save_era_labels()` to insert new data
3. This created a race condition where:
   - Thread A deletes its old data
   - Thread B deletes its old data
   - Thread A inserts new data
   - Thread B tries to insert but hits a constraint violation

## Solution
Combined the delete and insert operations into a single atomic transaction:

1. Modified `save_era_labels()` in `db_hybrid.rs` to:
   - Start a database transaction
   - Delete old data within the transaction
   - Insert new data within the same transaction
   - Commit the transaction

2. Removed all separate calls to `delete_era_labels_for_signal()` from `main.rs`

## Key Changes

### db_hybrid.rs
```rust
// Old approach (separate operations)
delete_era_labels_for_signal(...)?;
save_era_labels(...)?;

// New approach (atomic transaction)
pub fn save_era_labels(...) -> Result<()> {
    let mut transaction = conn.transaction()?;
    
    // Delete old data
    transaction.execute("DELETE FROM {} WHERE ...", &[...])?;
    
    // Insert new data
    let mut writer = transaction.copy_in(&copy_sql)?;
    writer.write_all(&csv_data)?;
    writer.finish()?;
    
    // Commit atomically
    transaction.commit()?;
}
```

### main.rs
Removed all calls to `delete_era_labels_for_signal()` since deletion is now handled inside the transaction.

## Benefits
1. **Atomicity**: Delete and insert happen as one unit
2. **No race conditions**: Each signal's transaction is independent
3. **Better performance**: Fewer round trips to the database
4. **Cleaner code**: Single function call instead of two

## Testing
Run the era detection again. It should now process all signals without duplicate key errors.