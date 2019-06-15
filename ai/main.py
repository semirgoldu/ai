from query_classifier import query_classifier as qc
while True:
	query=raw_input("Query: ")
	result= qc.predict(query)
	print result