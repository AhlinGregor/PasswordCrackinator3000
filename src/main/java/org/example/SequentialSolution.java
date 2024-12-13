package org.example;

public class SequentialSolution {
    private static String smallAlpha = "abcdefghijklmnopqrstuvwxyz";
    private static String bigAlpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private static char[] nonAlphabeticalCharacters = {
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  // Digits
            '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',   // Symbols
            '-', '_', '=', '+', '[', ']', '{', '}', '\\', '|',  // Brackets and slashes
            ';', ':', '\'', '\"', ',', '<', '.', '>', '/', '?', // Punctuation
            '`', '~'                                           // Miscellaneous
    };
    private static String nonAlphaString = new String(nonAlphabeticalCharacters);

    public static String computeDizShiz(String hash, boolean md5, int opt, int length) {
        String available;
        switch (opt) {
            case 1:
                available = SequentialSolution.smallAlpha;
                break;
                case 2:
                    available = SequentialSolution.bigAlpha;
                    break;
                    case 3:
                        available = SequentialSolution.nonAlphaString;
                        break;
                        case 4:
                            available = SequentialSolution.smallAlpha + SequentialSolution.bigAlpha;
                            break;
        }
        if (md5) {
            for (int i = 0; i < length; i++) {
            }
            //byte[] bytesOfMessage = text.getBytes("UTF-8");
        }
        return null;
    }
}
