import moveSchema_pb2  # Import the generated code

# Create an instance of the StringList message
string_list = moveSchema_pb2.MoveList()

# Add strings to the list
string_list.values.extend(["apple", "banana", "cherry"])

# Serialize the list to a binary format
serialized_data = string_list.SerializeToString()

print(f"Serialized Data: {serialized_data}")

# Deserialize the binary data back into a StringList object
deserialized_list = moveSchema_pb2.MoveList()
deserialized_list.ParseFromString(serialized_data)

print("Deserialized Data:", deserialized_list.values)
