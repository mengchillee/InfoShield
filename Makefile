demo:
	@echo "Running sample file..."
	@python3 infoshield.py data/sample_input.csv
	@echo " "
	@echo "Results (cluster label='-1' means 'unique')"
	@cat data/sample_input_full_LSH_labels.csv
	@echo "-----"
	@echo " "
	@echo "for colorful templates, check results/*/*/*.docx"
	@echo "  on mac: bash\% open results/*/*/*.docx"

clean:
	\rm -rf __pycache__
	\rm -rf results
	\rm -f data/sample_input_LSH_labels.csv
	\rm -f data/sample_input_full_LSH_labels.csv
	\rm -rf pkl_files
	\rm -f compression_rate.csv template_table.csv

spotless: clean
