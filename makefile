install:
	pip install -r requirements.txt

test:
	clear && python3 -m actions.test

run:
	clear && python3 -m actions.run

push:
	@echo "Commit message?"; \
	read msg; \
	git add -A; \
	git commit -m "$$msg"; \
	git push origin main