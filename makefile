install:
	pip install -r requirements.txt

test:
	clear && python3 -m actions.test_pipeline

assemble:
	clear && python3 -m actions.assemble_pipeline

complete:
	clear && python3 -m actions.complete_run

push:
	@echo "Commit message?"; \
	read msg; \
	git add -A; \
	git commit -m "$$msg"; \
	git push origin main