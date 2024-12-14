package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class GrafikaSeq {
    public static void createAndShowGUI() {
        // Create the main frame
        JFrame frame = new JFrame("GrafikaSeq GUI");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 400);

        // Create a panel to hold all components
        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(6, 1)); // 6 rows for the components

        // Add a text input field
        JTextField textField = new JTextField(20);
        panel.add(new JLabel("Enter text:"));
        panel.add(textField);

        // Add radio buttons
        JRadioButton md5 = new JRadioButton("MD-5");
        JRadioButton sha = new JRadioButton("SHA-256");
        ButtonGroup radioGroup = new ButtonGroup(); // Group the radio buttons
        radioGroup.add(md5);
        radioGroup.add(sha);

        JPanel radioPanel = new JPanel();
        radioPanel.add(md5);
        radioPanel.add(sha);
        panel.add(new JLabel("Select an option:"));
        panel.add(radioPanel);

        // Add checkboxes
        JCheckBox small = new JCheckBox("a-z");
        JCheckBox big = new JCheckBox("A-Z");
        JCheckBox nonAlpha = new JCheckBox("special");

        JPanel checkboxPanel = new JPanel();
        checkboxPanel.add(small);
        checkboxPanel.add(big);
        checkboxPanel.add(nonAlpha);
        panel.add(new JLabel("Select options:"));
        panel.add(checkboxPanel);

        // Add an integer menu
        JComboBox<Integer> dolzina = new JComboBox<>(new Integer[]{1, 2, 3, 4, 5});
        panel.add(new JLabel("Length:"));
        panel.add(dolzina);

        // Add a button
        JButton button = new JButton("Submit");
        button.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // Get text from the text field
                String text = textField.getText();

                // Get selected radio button
                boolean md5Selected = md5.isSelected();

                // Get selected checkboxes
                int selectedCheckboxes = 0;
                if (small.isSelected()) {
                    selectedCheckboxes += 1;
                }
                if (big.isSelected()) {
                    selectedCheckboxes += 2;
                }
                if (nonAlpha.isSelected()) {
                    selectedCheckboxes += 4;
                }

                // Get selected integer from the combo box
                int selectedInteger = (int) dolzina.getSelectedItem();

                long start = System.currentTimeMillis();
                String resitev = SequentialSolution.computeDizShiz(text, md5Selected, selectedCheckboxes, selectedInteger);
                long end = System.currentTimeMillis();
                System.out.println("Resitev je: \'" + resitev + "\' in porablo je " + (end - start) + "ms");
            }
        });
        panel.add(button);

        // Add the panel to the frame
        frame.add(panel);

        // Set the frame to be visible
        frame.setVisible(true);
    }
}
