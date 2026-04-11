from setuptools import setup, find_packages

setup(
    name="ev_charging_env",
    version="1.0.0",
    description="EV Charging Scheduler Environment",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "pytest>=7.0",
        "gradio>=3.50.0",
        "python-dotenv>=0.19.0"
    ],
)
