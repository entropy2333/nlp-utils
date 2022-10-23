.PHONY: modified_only_fixup all_fixup fixup

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = nlp_utils

check_dirs := nlp_utils

modified_only_fixup:
	$(eval modified_py_files := $(shell python scripts/get_modified_py_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		black --preview $(modified_py_files); \
		isort $(modified_py_files); \
		flake8 $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi

all_fixup:
	black --preview $(check_dirs)
	isort $(check_dirs)
	flake8 $(check_dirs)

# Format source code automatically and check is there are any problems left that need manual fixing

# Super fast fix and check target that only works on relevant modified files since the branch was made

fixup: modified_only_fixup
