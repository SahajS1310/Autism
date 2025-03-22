import mysql.connector


def create_connection():
    try:
        # Create a connection object
        conn = mysql.connector.connect(
            host='admin.cthasltjcbdq.ap-south-1.rds.amazonaws.com',
            port=3306,
            database='data',
            user='admin',
            password='admin12345'
        )

        return conn
    except Exception as e:
        print("Error: {}".format(e))
        raise Exception


def insert_data_into_table(data):
    # Connect to the database
    conn = create_connection()

    try:
        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()

        # Insert data into the table
        insert_data_query = '''
        INSERT INTO child_information (
            age_of_mother,
            age_of_father,
            medication_during_pregnancy,
            mode_of_delivery,
            health_of_mother,
            neurodevelopmental_condition,
            cognition_memory_level,
            anxiety_level,
            speech_level,
            social_interaction_communication,
            child_pays
        )  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);

        '''

        # Extract values from the JSON object
        values = (
            data['Age of mother while conceiving'],
            data['Age of Father at that time'],
            data['medication of mother during pregnancy, If any'],
            data['Mode of delivery'],
            data['Health of the mother while Conceiving and during pregnancy'],
            data['Any neurodevelopmental condition present or specially abled member in the family/ family history.'],
            data['Cognition and memory level of the child'],
            data['Anxiety level of child'],
            data['Speech level of child'],
            data['Social interaction/Communication of the child'],
            data['How much attention does the child pays']
        )

        print(values)
        # Execute the query
        cursor.execute(insert_data_query, values)

        # Commit the changes to the database
        conn.commit()

        # Close the cursor
        cursor.close()

        # Close the connection
        conn.close()

    except Exception as e:
        print("Error: {}".format(e))
        raise Exception
