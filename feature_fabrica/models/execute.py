from pydantic import BaseModel, Field


class ExecutionConfig(BaseModel):
    parallel_execution: bool = Field(default=False)
    max_workers: int = Field(default=0, ge=0)

def get_execution_config(parallel_execution: bool = False, max_workers: int = 0, reset_params: bool = False):
    if 'execution_config' not in globals() or reset_params:
        globals()['execution_config'] = ExecutionConfig(parallel_execution=parallel_execution, max_workers=max_workers)
    return globals()['execution_config']
