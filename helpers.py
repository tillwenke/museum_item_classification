# collects all column names that start with col_start
def col_collection(data, col_start):
        cols = []
        for c in data.columns:
            if (c.startswith(col_start)):
                cols.append(c)
        return cols