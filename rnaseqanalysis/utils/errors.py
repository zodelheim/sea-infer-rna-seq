from rich.traceback import install

install(show_locals=False)


class ConverterError(Exception):
    """Base converter error."""


class UnsupportedFormatError(ConverterError):
    """Unsupported input format."""


class InvalidDatasetError(ConverterError):
    """Dataset does not satisfy requirements."""


class MissingColumnError(InvalidDatasetError):
    """Required column is missing from adata.obs or adata.var."""

    def __init__(self, column: str, dataset: str, available: list[str], setby: str):
        self.column = column
        self.dataset = dataset
        self.available = available
        self.setby = setby
        super().__init__(
            f"Column '{column}', set by '{setby}' not found in {dataset}. \nAvailable columns: {available}"
        )
