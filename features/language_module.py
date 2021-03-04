from langdetect import detect
from langdetect import detect_langs
import dill

class LanguageDetector():
    
    def detect_language(self, lyrics):
        """
        Function to predict the most probable language
        for the given, unprocessed lyrics

        :param lyrics: Raw input lyrics (str)
        :return: Identificator for the language (ISO 639-1 codes) (str)
        """

        try:
            return detect(lyrics)
        except Exception:
            return None


    def detect_languages(self, lyrics):
        """
        Function to predict the most probable language
        for the given, unprocessed lyrics

        :param lyrics: Raw input lyrics (str)
        :return: 
            languages: List containing the identificator for the language (ISO 639-1 codes)
            probabilities: List containing the identification probabilites
        """

        try:
            # Returns list including the languages and the probabilities
            detected_languages = detect_langs(lyrics)

            # Extract the languages and probabilites into two seperate lists
            languages = [detected.lang for detected in detected_languages]
            probabilities = [detected.prob for detected in detected_languages]

            return languages, probabilities

        except Exception:
            return None

'''
detector = LanguageDetector()

string = "Schaut nicht weg, wenn das Elend plagt"
print(detector.detect_language(string))
print(detector.detect_languages(string))
'''