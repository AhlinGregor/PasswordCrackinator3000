package org.example;

import javax.swing.*;
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
    private final static String nonAlphaString = new String(nonAlphabeticalCharacters);

    // Can I offer you a static method in these trying times

    public static String computeDizShiz(String hash, boolean md5, int opt, int length, JProgressBar progressBar, long totalCombinations) {
        String available = getCharacterSet(opt);
        if (md5) {
            if (!isValidMD5(hash)) return null;

            // Big daddy method for md5
            return findMatchingPermutationMD(hash, available, length, progressBar, totalCombinations);
        } else {
            if (!isValidSHA(hash)) return null;

            // Big daddy method for sha-256
            return findMatchingPermutationSHA(hash, available, length, progressBar, totalCombinations);
        }
    }

    private static boolean isValidSHA(String input) {
        return input != null && input.matches("^[a-fA-F0-9]{64}$"); //"Yayy! Regex!" said he, sarcastically
    }

    public static long calculateTotalCombinations(int charsetLength, int maxLength) {

        return (long) Math.pow(charsetLength, maxLength);
    }

    private static String findMatchingPermutationSHA(String hash, String available, int maxLength, JProgressBar progressBar, long totalCombinations) {
        return findMatchingPermutationSHAHelper(hash, available, "", maxLength, progressBar, totalCombinations, new long[]{0});
    }

    private static String findMatchingPermutationSHAHelper(String hash, String str, String prefix, int maxLength, JProgressBar progressBar, long totalCombinations, long[] currentProgress) {
        // Base case: Check if prefix matches the hash
        if (prefix.length() == maxLength) {
            currentProgress[0]++;
            SwingUtilities.invokeLater(() -> {
                progressBar.setValue((int) currentProgress[0]);
                progressBar.setString(currentProgress[0] + "/" + totalCombinations);
            });

            try {
                MessageDigest md = MessageDigest.getInstance("SHA-256");
                byte[] digest = md.digest(prefix.getBytes(StandardCharsets.UTF_8));
                StringBuilder sb = new StringBuilder();
                for (byte b : digest) {
                    //sb.append(String.format("%02x", b));
                    String hex = Integer.toHexString(0xff & b);
                    if (hex.length() == 1) {
                        sb.append('0');
                    }
                    sb.append(hex);
                }
                if (sb.toString().equals(hash)) {
                    progressBar.setValue((int) currentProgress[0]); // If success set pb to 100%
                    progressBar.setString(currentProgress[0] + "/" + totalCombinations);
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

            // Forgive me Vicic for i have sinned
            String result = findMatchingPermutationSHAHelper(hash, str, prefix + str.charAt(i), maxLength, progressBar, totalCombinations, currentProgress);
            if (result != null) {
                return result; // Early stopping
            }
        }
        return null;
    }

    private static String findMatchingPermutationMD(String hash, String available, int maxLength, JProgressBar progressBar, long totalCombinations) {
        return findMatchingPermutationMDHelper(hash, available, "", maxLength, progressBar, totalCombinations, new long[]{0});
    }

    private static String findMatchingPermutationMDHelper(String hash, String str, String prefix, int maxLength, JProgressBar progressBar, long totalCombinations, long[] currentProgress) {
        // Base case: Check if prefix matches the hash
        if (prefix.length() == maxLength) {
            currentProgress[0]++;
            SwingUtilities.invokeLater(() -> {
                progressBar.setValue((int) currentProgress[0]);
                progressBar.setString(currentProgress[0] + "/" + totalCombinations);
            });

            try {
                MessageDigest md = MessageDigest.getInstance("MD5");
                byte[] digest = md.digest(prefix.getBytes(StandardCharsets.UTF_8));
                StringBuilder sb = new StringBuilder();
                for (byte b : digest) {
                    sb.append(String.format("%02x", b));
                }
                if (sb.toString().equals(hash)) {
                    progressBar.setValue((int) currentProgress[0]); // If success set pb to 100%
                    progressBar.setString(currentProgress[0] + "/" + totalCombinations);
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

            // I apologise for NOTHING
            String result = findMatchingPermutationMDHelper(hash, str, prefix + str.charAt(i), maxLength, progressBar, totalCombinations, currentProgress);
            if (result != null) {
                return result; // Early stopping
            }
        }
        return null;
    }

    private static boolean isValidMD5(String input) {
        if (input == null || input.length() != 32) {
            return false;
        }

        return input.matches("[a-fA-F0-9]{32}"); // The bane of my existence yet again
    }

    public static String dictionaryAttack(File file, String hash, boolean md5) {
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (md5) {
                    String lineHash = computeMD5Hash(line);
                    if (lineHash.equals(hash)) return line;
                } else {
                    String lineHash = computeSHA256Hash(line);
                    if (lineHash.equals(hash)) return line;
                }
            }
        } catch (IOException ex) {
            System.err.println("Error reading the file: " + ex.getMessage());
        }
        return null;
    }

    private static String computeSHA256Hash(String input) {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = md.digest(input.getBytes());
            return byteArrayToHexString(hashBytes);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
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

    public static String getCharacterSet(int opt) {
        return switch (opt) {
            case 1 -> SequentialSolution.smallAlpha;
            case 2 -> SequentialSolution.bigAlpha;
            case 3 -> SequentialSolution.smallAlpha + SequentialSolution.bigAlpha;
            case 4 -> SequentialSolution.nonAlphaString;
            case 5 -> SequentialSolution.smallAlpha + SequentialSolution.nonAlphaString;
            case 6 -> SequentialSolution.bigAlpha + SequentialSolution.nonAlphaString;
            default -> SequentialSolution.smallAlpha + SequentialSolution.bigAlpha + SequentialSolution.nonAlphaString;
        };
    }
}
