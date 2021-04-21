"errand setup module."

def main():

    from setuptools import setup, find_packages
    from errand.main import Errand as erd

    console_scripts = ["errand=errand.__main__:main"]
    install_requires = ["numpy"]

    setup(
        name=erd._name_,
        version=erd._version_,
        description=erd._description_,
        long_description=erd._long_description_,
        author=erd._author_,
        author_email=erd._author_email_,
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        keywords="errand",
        packages=find_packages(exclude=["tests"]),
        include_package_data=True,
        install_requires=install_requires,
        entry_points={ "console_scripts": console_scripts},
        project_urls={
            "Bug Reports": "https://github.com/grnydawn/errand/issues",
            "Source": "https://github.com/grnydawn/errand",
        }
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
