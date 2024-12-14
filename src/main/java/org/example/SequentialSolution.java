package org.example;

import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;

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
    public static String nonAlphaString = new String(nonAlphabeticalCharacters);

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
                available = SequentialSolution.smallAlpha + SequentialSolution.bigAlpha;
                break;
            case 4:
                available = SequentialSolution.nonAlphaString;
                break;
            case 5:
                available = SequentialSolution.smallAlpha + SequentialSolution.nonAlphaString;
                break;
            case 6:
                available = SequentialSolution.bigAlpha + SequentialSolution.nonAlphaString;
                break;
            default:
                available = SequentialSolution.smallAlpha + SequentialSolution.bigAlpha + SequentialSolution.nonAlphaString;
                break;
        }

        ArrayList<String> permutacije = generatePermutations(available, length);

        if (md5) {
            for (int i = 0; i < permutacije.size(); i++) {
                byte[] bytesOfMessage = null;
                try {
                    bytesOfMessage = permutacije.get(i).getBytes("UTF-8");
                } catch (UnsupportedEncodingException e) {
                    throw new RuntimeException(e);
                }

                MessageDigest md = null;
                try {
                    md = MessageDigest.getInstance("MD5");
                } catch (NoSuchAlgorithmException e) {
                    throw new RuntimeException(e);
                }
                byte[] theMD5digest = md.digest(bytesOfMessage);
                String rez = new String(theMD5digest);

                if (rez.equals(hash)) return permutacije.get(i);
            }
        }
        return null;
    }

    /**
     * Generates all permutations of a string from length 1 to maxLength, allowing repetition of characters.
     * @param str The input string
     * @param maxLength The maximum length of permutations to generate
     * @return A list of all permutations
     */
    public static ArrayList<String> generatePermutations(String str, int maxLength) {
        ArrayList<String> result = new ArrayList<>();
        generatePermutationsHelper(str, "", result, maxLength);
        return result;
    }

    /**
     * Helper method to recursively generate permutations with repeated characters allowed.
     * @param str The original character set
     * @param prefix The current prefix (partial permutation)
     * @param result The list to store results
     * @param maxLength The maximum length of permutations
     */
    private static void generatePermutationsHelper(String str, String prefix, ArrayList<String> result, int maxLength) {
        // Add the current prefix to the result if its length is within the range
        if (!prefix.isEmpty() && prefix.length() <= maxLength) {
            result.add(prefix);
        }

        // Base case: Stop recursion if prefix length reaches maxLength
        if (prefix.length() >= maxLength) {
            return;
        }

        // Recursive case: Iterate over all characters, allowing repetition
        for (int i = 0; i < str.length(); i++) {
            char ch = str.charAt(i); // Current character
            generatePermutationsHelper(str, prefix + ch, result, maxLength); // Allow repetition by passing the full string
        }
    }
}
