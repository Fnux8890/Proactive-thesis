## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Tested on both CPU and GPU backends (if applicable)

## Resource Lifetime Checklist
- [ ] Generators own their connection (`chunked_query` pattern)
- [ ] All database connections are properly closed
- [ ] No connection leaks in error paths

## Backend Compatibility
- [ ] Uses `backend.pd` for DataFrame operations
- [ ] Used backend.dtypes helpers for type checking
- [ ] Explicit conversions noted with comments
- [ ] No raw `import pandas` or `import cudf` outside backend module
- [ ] Parquet writes use `index=False` for parity
- [ ] No f-string SQL filters (use parameterized queries)
- [ ] Vectorized sentinel replacement used

## External Libraries
- [ ] tsfresh adapter used (if applicable)
- [ ] sklearn adapter used (if applicable)
- [ ] All external library boundaries documented

## Code Quality
- [ ] Code follows project style guidelines
- [ ] Added/updated tests as needed
- [ ] Documentation updated
- [ ] No hardcoded values or credentials

## Performance
- [ ] Profiled for memory usage (if applicable)
- [ ] Batch operations used for database operations
- [ ] GPU acceleration tested (if applicable)

## Additional Notes
Any additional context or screenshots