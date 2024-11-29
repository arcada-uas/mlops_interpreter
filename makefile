install:
	pip install -r requirements.txt

test:
	clear && python3 -m actions.test_pipeline

create:
	clear && python3 -m actions.create_pipeline

train:
	clear && python3 -m actions.train_pipeline

push:
	@echo "Commit message?"; \
	read msg; \
	git add -A; \
	git commit -m "$$msg"; \
	git push origin main