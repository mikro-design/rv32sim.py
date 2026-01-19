test:
	python3 -m pytest

test-examples:
	$(MAKE) -C examples test

clean:
	$(MAKE) -C examples clean
