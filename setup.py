from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="ai-models-xihe",
    version="0.1.0",
    description="XiHe ocean forecasting adapter for ai-models and ONNX Runtime",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="European Centre for Medium-Range Weather Forecasts (ECMWF)",
    author_email="software.support@ecmwf.int",
    url="https://github.com/GexLoong/ai-models-xihe",
    project_urls={
        "Source": "https://github.com/GexLoong/ai-models-xihe",
        "Issues": "https://github.com/GexLoong/ai-models-xihe/issues",
    },
    packages=find_packages(include=["ai_models_xihe", "ai_models_xihe.*"]),
    package_data={"ai_models_xihe": ["data/*.yaml", "data/*.npy"]},
    python_requires=">=3.9",
    # ai-models 0.7 imports the pre-1.0 earthkit FieldArray API.
    install_requires=["ai-models>=0.7,<0.8", "earthkit-data>=0.20,<1", "numpy>=1.21,<2", "xarray", "pandas", "PyYAML", "netCDF4"],
    entry_points={
        "ai_models.model": ["xihe = ai_models_xihe.model:XiHe"],
        "ai_models.output": ["xihe-netcdf = ai_models_xihe._common:NetCDFOutput"],
    },
)
