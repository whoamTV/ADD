def repair_v2_0_0(database_list):
    test_data_logger = database_list.get("test_data_logger")
    cursor = test_data_logger.conn.cursor()
    cursor.execute("ALTER TABLE attack_dimension_summary add column ACAMC_T float")
    cursor.execute("ALTER TABLE attack_dimension_summary add column ACAMC_A float")
    cursor.execute("ALTER TABLE attack_dimension_summary add column TAS float")
    cursor.execute("ALTER TABLE attack_dimension_summary add column OTR_MR float")
    cursor.execute("ALTER TABLE attack_dimension_summary add column OTR_AIAC float")
    cursor.execute("ALTER TABLE attack_dimension_summary add column OTR_ARTC float")
    cursor.close()
    test_data_logger.conn.commit()