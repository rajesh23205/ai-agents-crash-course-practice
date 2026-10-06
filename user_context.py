class Student:
    def __init__(self, name: str, experience_level: str, favorite_language: str):
        self.name = name
        self.experience_level = experience_level
        self.favorite_language = favorite_language

    def __str__(self) -> str:
        """Returns a clean string representation of the student."""
        return (
            f"Student Name: {self.name}\n"
            f"Experience Level: {self.experience_level}\n"
            f"Favorite Language: {self.favorite_language}"
        )


if __name__ == "__main__":
    # Create the student instance for Rajesh
    student_1 = Student(
        name="Rajesh", experience_level="beginner", favorite_language="Python"
    )

    # Print the student details
    print(student_1)
