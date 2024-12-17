package org.example;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Formatter;

public class SequentialSolution {
    private final static String smallAlpha = "abcdefghijklmnopqrstuvwxyz";
    private final static String bigAlpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private final static char[] nonAlphabeticalCharacters = {
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  // Digits
            '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',   // Symbols
            '-', '_', '=', '+', '[', ']', '{', '}', '\\', '|',  // Brackets and slashes
            ';', ':', '\'', '\"', ',', '<', '.', '>', '/', '?', // Punctuation
            '`', '~'                                           // Miscellaneous
    };
    public static String nonAlphaString = new String(nonAlphabeticalCharacters);


    public static String computeDizShiz(String hash, boolean md5, int opt, int length) {
        if (md5) {
            if (!isValidMD5(hash)) {
                return null;
            }

            String available = switch (opt) {
                case 1 -> SequentialSolution.smallAlpha;
                case 2 -> SequentialSolution.bigAlpha;
                case 3 -> SequentialSolution.smallAlpha + SequentialSolution.bigAlpha;
                case 4 -> SequentialSolution.nonAlphaString;
                case 5 -> SequentialSolution.smallAlpha + SequentialSolution.nonAlphaString;
                case 6 -> SequentialSolution.bigAlpha + SequentialSolution.nonAlphaString;
                default -> SequentialSolution.smallAlpha + SequentialSolution.bigAlpha + SequentialSolution.nonAlphaString;
            };

            // Generate permutations dynamically to avoid memory overhead
            return findMatchingPermutation(hash, available, length);
        }
        else return null;
    }

    private static String findMatchingPermutation(String hash, String available, int maxLength) {
        return findMatchingPermutationHelper(hash, available, "", maxLength);
    }

    private static String findMatchingPermutationHelper(String hash, String str, String prefix, int maxLength) {
        // Base case: Check if prefix matches the hash
        if (!prefix.isEmpty() && prefix.length() <= maxLength) {
            try {
                MessageDigest md = MessageDigest.getInstance("MD5");
                byte[] digest = md.digest(prefix.getBytes(StandardCharsets.UTF_8));
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

        // Stop recursion if prefix length exceeds maxLength
        if (prefix.length() >= maxLength) {
            return null;
        }

        // Recursive case: Generate permutations dynamically
        for (int i = 0; i < str.length(); i++) {
            String result = findMatchingPermutationHelper(hash, str, prefix + str.charAt(i), maxLength);
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

    public static String dictionaryAttack(File file, String hash, boolean md5) {
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (md5) {
                    String lineHash = computeMD5Hash(line);
                    if (lineHash.equals(hash)) return line;
                }
            }
        } catch (IOException ex) {
            System.err.println("Error reading the file: " + ex.getMessage());
        }
        return null;
    }

    // Helper method to convert a byte array to a hex string
    private static String byteArrayToHexString(byte[] bytes) {
        Formatter formatter = new Formatter();
        for (byte b : bytes) {
            formatter.format("%02x", b);
        }
        String hexString = formatter.toString();
        formatter.close();
        return hexString;
    }

    // Helper method to compute the MD5 hash of a given string
    private static String computeMD5Hash(String input) {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("MD5");
            byte[] hashBytes = md.digest(input.getBytes());
            return byteArrayToHexString(hashBytes);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

}
