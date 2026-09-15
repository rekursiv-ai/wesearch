from typing import Literal

type HttpVersionLiteral = Literal[
    "v1",
    "v2",
    "v2tls",
    "v2_prior_knowledge",
    "v3",
    "v3only",
]

class CurlCffiWarning(UserWarning, RuntimeWarning): ...

def config_warnings(on: bool = ...) -> None: ...
