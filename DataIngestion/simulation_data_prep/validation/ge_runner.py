from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict
import importlib.metadata as _im

import polars as pl
import great_expectations as gx
from prefect import task
from prefect.logging import get_run_logger
from prefect.exceptions import MissingContextError


def _modern_gx() -> bool:
    """Check if the installed Great Expectations version is 0.18.0 or higher."""
    ver = _im.version("great_expectations")
    major, minor, *_ = map(int, ver.split(".")[:2])
    return (major, minor) >= (0, 18)


def get_batch_request(
    context: gx.DataContext,
    ds_name: str,
    asset_name: str,
    opts: Dict[str, Any],
) -> Any: # gx.batch_request.BatchRequest type varies across versions
    """Builds a BatchRequest compatible with both legacy and modern GX APIs for runtime data."""
    if _modern_gx():
        # Modern GX (>= 0.18): Instantiate RuntimeBatchRequest directly
        try:
            # The exact import path might vary slightly in very recent versions, but core.batch is standard
            from great_expectations.core.batch import RuntimeBatchRequest
        except ImportError:
            # Fallback for potential future structure changes or if namespace changed
            try:
                from gx.core.batch import RuntimeBatchRequest
            except ImportError:
                logger = _logger() # Assuming _logger is available in this scope or defined globally
                logger.error("Could not import RuntimeBatchRequest from expected locations.", exc_info=True)
                raise

        batch_data = opts.get("batch_data")
        if batch_data is None:
             raise ValueError("batch_data must be provided in opts for RuntimeBatchRequest")

        # For runtime data, we typically define the connector name when adding the datasource
        # Ensure this connector name matches how the datasource was configured.
        data_connector_name = "runtime_connector" # Hardcoded based on the add_datasource call

        # Construct the RuntimeBatchRequest instance
        # Include batch_identifiers if needed, otherwise default is usually fine for runtime
        batch_identifiers = {"run_id": f"runtime_{datetime.utcnow().isoformat()}"} # Example identifier

        return RuntimeBatchRequest(
            datasource_name=ds_name,
            data_connector_name=data_connector_name,
            data_asset_name=asset_name, # Creates a temporary asset name
            runtime_parameters={"batch_data": batch_data},
            batch_identifiers=batch_identifiers,
        )
    else:
        # Legacy GX (< 0.18): Use the get_batch_request method on the Datasource object
        ds = context.get_datasource(ds_name)
        # Pass batch_data directly for runtime requests in older versions
        batch_data = opts.get("batch_data")
        if batch_data is not None:
            return ds.get_batch_request(data_asset_name=asset_name, batch_data=batch_data)
        else:
             # Fallback if batch_data wasn't passed explicitly in opts, though it should be for runtime
            return ds.get_batch_request(data_asset_name=asset_name, options=opts)


def _logger() -> logging.Logger:
    try:
        return get_run_logger()
    except MissingContextError:
        # Ensure logger setup if called outside Prefect run
        lg = logging.getLogger("ge_runner_helper")
        if not lg.handlers:
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             lg.addHandler(handler)
             lg.setLevel(logging.INFO) # Or your desired level
        return lg


# Keep as Prefect task if needed by flow, or remove decorator for standalone use
@task
def validate_with_ge(
    df: pl.DataFrame,
    ge_root_dir: str = "/app/great_expectations",
    skip_validation: bool = False,
) -> bool:
    """Validates the transformed DataFrame using Great Expectations."""
    logger = _logger()

    if skip_validation:
        logger.info("Great Expectations validation explicitly skipped by configuration.")
        return True # Treat skipped validation as success

    logger.info("Starting data validation with Great Expectations...")
    if df.is_empty():
        logger.warning("DataFrame is empty, skipping validation.")
        return True # Or False? Decide based on requirements - returning True for now

    suite_name = "greenhouse_features"
    datasource_name = "runtime_pandas_datasource"
    data_asset_name = f"features_{datetime.utcnow().date().isoformat()}"

    try:
        logger.info(f"Initializing Great Expectations DataContext at {ge_root_dir}")
        context = gx.get_context(context_root_dir=ge_root_dir)

        try:
            context.get_datasource(datasource_name)
            logger.info(f"Using existing GE datasource: {datasource_name}")
        except gx.exceptions.DatasourceError:
            logger.info(f"Adding runtime datasource '{datasource_name}' to GE context")
            context.add_datasource(
                name=datasource_name,
                class_name="Datasource",
                execution_engine={"class_name": "PandasExecutionEngine"},
                data_connectors={
                    "runtime_connector": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["run_id"],
                    }
                },
            )
            logger.info(f"Added runtime datasource: {datasource_name}")

        logger.info("Preparing GE Runtime Batch Request...")
        pandas_df = df.to_pandas()
        # Use the helper function to create the batch request
        # Pass the data directly within the options dict for the helper
        batch_request_options = {"batch_data": pandas_df}
        batch_request = get_batch_request(
            context,
            datasource_name,
            data_asset_name,
            batch_request_options
        )

        logger.info(f"Running validation against suite: {suite_name}")
        # Use run_checkpoint for runtime data to avoid saving data in the CheckpointStore
        # The structure for validations is often needed here too.
        result = context.run_checkpoint(
            validations=[{
                "batch_request": batch_request,
                "expectation_suite_name": suite_name
            }]
            # For older versions (<0.18), the signature might differ slightly
            # e.g., might need batch_request=batch_request, expectation_suite_name=suite_name
            # but since we are pinned to 0.18.5, the 'validations' structure is correct.
        )

        success = bool(result.success)
        if success:
            logger.info("Great Expectations validation PASSED.")
        else:
            logger.error("Great Expectations validation FAILED.")
            try:
                stats = result.run_results[next(iter(result.run_results))]["validation_result"]["statistics"]
                logger.error(f"  Validation Stats: {stats}")
                failed_expectations = [
                    res for res in result.list_validation_results() if not res["success"]
                ]
                if failed_expectations:
                    logger.error("  Failed Expectations:")
                    for failure in failed_expectations:
                        exp = failure["expectation_config"]
                        details = failure["result"]
                        logger.error(
                            f"    - {exp['expectation_type']}({exp['kwargs']}): "
                            f"{details.get('unexpected_list', details.get('partial_unexpected_list', details.get('observed_value', '')))}"
                        )
            except Exception as log_e:
                logger.error(f"  Error logging GE failure details: {log_e}")

        return success

    except Exception as e:
        logger.error(f"Error during Great Expectations validation: {e}", exc_info=True)
        return False 