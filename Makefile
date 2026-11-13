.PHONY: install ingest ask eval test app
install:
	pip install -r requirements.txt
ingest:
	python -m docchat ingest
ask:
	python -m docchat ask "How many days of annual leave do I get?"
eval:
	python -m docchat eval
test:
	pytest -q
app:
	streamlit run app/streamlit_app.py
