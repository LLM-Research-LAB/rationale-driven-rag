# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="tests.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

# Test messages
logger.info("Starting Logs!")
