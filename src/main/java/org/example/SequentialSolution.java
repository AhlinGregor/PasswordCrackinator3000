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
        if (!isValidMD5(hash)) {
            return null;
        }

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

        // Generate permutations dynamically to avoid memory overhead
        return findMatchingPermutation(hash, available, md5, length);
    }

    private static String findMatchingPermutation(String hash, String available, boolean md5, int maxLength) {
        return findMatchingPermutationHelper(hash, available, "", md5, maxLength);
    }

    private static String findMatchingPermutationHelper(String hash, String str, String prefix, boolean md5, int maxLength) {
        // Base case: Check if prefix matches the hash
        if (!prefix.isEmpty() && prefix.length() <= maxLength) {
            if (md5) {
                try {
                    MessageDigest md = MessageDigest.getInstance("MD5");
                    byte[] digest = md.digest(prefix.getBytes("UTF-8"));
                    StringBuilder sb = new StringBuilder();
                    for (byte b : digest) {
                        sb.append(String.format("%02x", b));
                    }
                    if (sb.toString().equals(hash)) {
                        return prefix;
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }

        // Stop recursion if prefix length exceeds maxLength
        if (prefix.length() >= maxLength) {
            return null;
        }

        // Recursive case: Generate permutations dynamically
        for (int i = 0; i < str.length(); i++) {
            String result = findMatchingPermutationHelper(hash, str, prefix + str.charAt(i), md5, maxLength);
            if (result != null) {
                return result; // Early stopping
            }
        }
        return null;
    }

    public static boolean isValidMD5(String input) {
        if (input == null || input.length() != 32) {
            return false;
        }

        return input.matches("[a-fA-F0-9]{32}");
    }
}
