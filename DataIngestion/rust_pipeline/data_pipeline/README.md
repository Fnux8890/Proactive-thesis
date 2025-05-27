# Enhanced Parallel Rust Data Pipeline

This is a high-performance data ingestion pipeline built with Rust, featuring parallel processing capabilities using Rayon and Tokio.

## Key Enhancements

### 1. **Parallel File Processing**
- Uses Rayon for CPU-bound file parsing operations
- Processes multiple files concurrently with configurable worker threads
- Automatic load balancing across available CPU cores

### 2. **Parallel Database Operations**
- Batch inserts with configurable batch size (default: 5000 records)
- Concurrent batch processing with semaphore-controlled parallelism
- Connection pooling for efficient database resource usage

### 3. **Retry Mechanisms**
- Exponential backoff retry for transient failures
- Configurable retry policies for different operation types
- Separate retry configurations for file and database operations

### 4. **Performance Metrics**
- Real-time tracking of processing statistics
- Detailed timing information for each pipeline stage
- Throughput calculation and reporting

### 5. **Progress Tracking**
- Visual progress bars for file processing
- Detailed logging at each stage
- Failed file tracking and reporting

## Architecture

```
┌─────────────────┐
│ Configuration   │
│ (data_files.json)│
└────────┬────────┘
         │
         v
┌─────────────────┐     ┌──────────────────┐
│ Glob Expansion  │────>│ Parallel File    │
│ (Parallel)      │     │ Processing       │
└─────────────────┘     │ (Rayon)          │
                        └────────┬─────────┘
                                 │
                                 v
                        ┌──────────────────┐
                        │ Record           │
                        │ Collection       │
                        └────────┬─────────┘
                                 │
                                 v
                        ┌──────────────────┐
                        │ Parallel DB      │
                        │ Insertion        │
                        │ (Tokio + Batches)│
                        └────────┬─────────┘
                                 │
                                 v
                        ┌──────────────────┐
                        │ Validation &     │
                        │ Merge            │
                        └──────────────────┘
```

## Module Structure

- `main.rs` - Pipeline orchestration and entry point
- `parallel.rs` - Parallel processing utilities
- `db_operations.rs` - Database operations with batching
- `retry.rs` - Retry logic with exponential backoff
- `metrics.rs` - Performance metrics tracking
- `file_processor.rs` - File parsing dispatch
- `parsers/` - Format-specific parsers
- `config.rs` - Configuration loading
- `validation.rs` - Data validation
- `errors.rs` - Error types

## Performance Optimizations

1. **Parallel Glob Expansion**: File patterns are expanded in parallel
2. **Work Stealing**: Rayon's work-stealing scheduler ensures optimal CPU utilization
3. **Batch Processing**: Records are inserted in configurable batches to reduce database round trips
4. **Connection Pooling**: Database connections are reused across operations
5. **Binary COPY**: Uses PostgreSQL's binary COPY for maximum insertion speed

## Configuration

The pipeline is configured via environment variables:

- `DATABASE_URL` - PostgreSQL connection string
- `RUST_LOG` - Log level (debug, info, warn, error)

## Building and Running

```bash
# Build in release mode for optimal performance
cargo build --release

# Run with environment variables
DATABASE_URL=postgresql://user:pass@host:port/db cargo run --release
```

## Monitoring

The pipeline provides comprehensive metrics including:
- Total files attempted/successful/failed
- Records parsed and inserted
- Processing times for each stage
- Overall throughput (records/sec)

## Error Handling

- Graceful error recovery with detailed logging
- Failed files are tracked and reported
- Retry logic for transient failures
- Transaction rollback on batch failures