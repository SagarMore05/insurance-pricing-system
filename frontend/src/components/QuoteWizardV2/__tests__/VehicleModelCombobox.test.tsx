import { describe, it, expect, vi } from 'vitest'
import { useState } from 'react'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { VehicleModelCombobox } from '../QuoteWizardV2'

// ─── Part 1: Searchable vehicle model combobox ───────────────────────────────

function renderCombobox(props: Partial<Parameters<typeof VehicleModelCombobox>[0]> = {}) {
  const defaults = {
    make: 'Maruti',
    value: '',
    onChange: vi.fn(),
    isCustom: false,
    onCustomToggle: vi.fn(),
    error: undefined,
  }
  return render(<VehicleModelCombobox {...defaults} {...props} />)
}

describe('VehicleModelCombobox — Part 1: vehicle model experience', () => {
  describe('disabled state — no make selected', () => {
    it('is disabled when make is empty', () => {
      renderCombobox({ make: '' })
      const input = screen.getByRole('textbox')
      expect(input).toBeDisabled()
    })

    it('shows placeholder "Select a make first" when no make', () => {
      renderCombobox({ make: '' })
      expect(screen.getByPlaceholderText('Select a make first')).toBeInTheDocument()
    })
  })

  describe('enabled state — with make selected', () => {
    it('is enabled when a make is provided', () => {
      renderCombobox({ make: 'Maruti' })
      const input = screen.getByRole('textbox')
      expect(input).not.toBeDisabled()
    })

    it('shows "Search or select model..." placeholder when make is set', () => {
      renderCombobox({ make: 'Mahindra' })
      expect(screen.getByPlaceholderText('Search or select model...')).toBeInTheDocument()
    })

    it('has aria-expanded attribute', () => {
      renderCombobox({ make: 'Maruti' })
      const input = screen.getByRole('textbox')
      expect(input).toHaveAttribute('aria-expanded')
    })

    it('has aria-label for accessibility', () => {
      renderCombobox({ make: 'Maruti' })
      expect(screen.getByLabelText('Vehicle model search')).toBeInTheDocument()
    })
  })

  describe('dropdown opens on focus', () => {
    it('shows models when input is focused', async () => {
      renderCombobox({ make: 'Maruti' })
      const input = screen.getByRole('textbox')
      await userEvent.click(input)
      // Maruti models should appear
      expect(screen.getByText('Swift')).toBeInTheDocument()
      expect(screen.getByText('Alto')).toBeInTheDocument()
      expect(screen.getByText('Baleno')).toBeInTheDocument()
    })

    it('always shows "Other / Model Not Listed" at bottom of list', async () => {
      renderCombobox({ make: 'Maruti' })
      const input = screen.getByRole('textbox')
      await userEvent.click(input)
      expect(screen.getByText('Other / Model Not Listed')).toBeInTheDocument()
    })
  })

  describe('type-to-search filtering', () => {
    it('filters models by query — "sw" matches Swift', async () => {
      renderCombobox({ make: 'Maruti' })
      const input = screen.getByRole('textbox')
      await userEvent.click(input)
      await userEvent.type(input, 'sw')
      expect(screen.getByText('Swift')).toBeInTheDocument()
      // Alto should not appear with query "sw"
      expect(screen.queryByText('Alto')).not.toBeInTheDocument()
    })

    it('shows "No models match" message for unknown query', async () => {
      renderCombobox({ make: 'Maruti' })
      const input = screen.getByRole('textbox')
      await userEvent.click(input)
      await userEvent.type(input, 'zzzzz')
      expect(screen.getByText(/No models match/)).toBeInTheDocument()
    })

    it('filtering is case-insensitive — "SWIFT" matches Swift', async () => {
      renderCombobox({ make: 'Maruti' })
      const input = screen.getByRole('textbox')
      await userEvent.click(input)
      await userEvent.type(input, 'SWIFT')
      expect(screen.getByText('Swift')).toBeInTheDocument()
    })
  })

  describe('model selection', () => {
    it('calls onChange with the selected model name', async () => {
      const onChange = vi.fn()
      renderCombobox({ make: 'Maruti', onChange })
      const input = screen.getByRole('textbox')
      await userEvent.click(input)
      await userEvent.click(screen.getByText('Swift'))
      expect(onChange).toHaveBeenCalledWith('Swift')
    })

    it('displays selected model value in input when closed', () => {
      renderCombobox({ make: 'Maruti', value: 'Swift' })
      const input = screen.getByRole('textbox')
      expect(input).toHaveValue('Swift')
    })
  })

  describe('"Other / Model Not Listed" — custom mode', () => {
    it('calls onCustomToggle(true) when "Other" is selected', async () => {
      const onCustomToggle = vi.fn()
      renderCombobox({ make: 'Maruti', onCustomToggle })
      const input = screen.getByRole('textbox')
      await userEvent.click(input)
      await userEvent.click(screen.getByText('Other / Model Not Listed'))
      expect(onCustomToggle).toHaveBeenCalledWith(true)
    })
  })

  describe('custom model mode (isCustom = true)', () => {
    it('shows amber advisory banner in custom mode', () => {
      renderCombobox({ make: 'Maruti', isCustom: true })
      expect(screen.getByText(/Custom model/)).toBeInTheDocument()
      expect(screen.getByText(/standard vehicle data will be used/)).toBeInTheDocument()
    })

    it('shows "Back to list" button in custom mode', () => {
      renderCombobox({ make: 'Maruti', isCustom: true })
      expect(screen.getByRole('button', { name: /Back to list/i })).toBeInTheDocument()
    })

    it('"Back to list" calls onCustomToggle(false) and onChange("")', async () => {
      const onCustomToggle = vi.fn()
      const onChange = vi.fn()
      renderCombobox({ make: 'Maruti', isCustom: true, onCustomToggle, onChange })
      await userEvent.click(screen.getByRole('button', { name: /Back to list/i }))
      expect(onCustomToggle).toHaveBeenCalledWith(false)
      expect(onChange).toHaveBeenCalledWith('')
    })

    it('shows a free-text input in custom mode for entering model name', () => {
      renderCombobox({ make: 'Maruti', isCustom: true })
      expect(screen.getByPlaceholderText(/Enter your vehicle model/)).toBeInTheDocument()
    })

    it('custom text input has accessible aria-label', () => {
      renderCombobox({ make: 'Maruti', isCustom: true })
      expect(screen.getByLabelText('Custom vehicle model')).toBeInTheDocument()
    })

    it('typing in custom mode updates the input value', async () => {
      // Use a stateful wrapper so the controlled input re-renders with each keystroke
      function ControlledCombobox() {
        const [val, setVal] = useState('')
        return (
          <VehicleModelCombobox
            make="Maruti" value={val} onChange={setVal}
            isCustom={true} onCustomToggle={vi.fn()}
          />
        )
      }
      render(<ControlledCombobox />)
      const input = screen.getByLabelText('Custom vehicle model')
      await userEvent.type(input, 'Baleno Sigma')
      expect(screen.getByDisplayValue('Baleno Sigma')).toBeInTheDocument()
    })
  })

  describe('error display', () => {
    it('shows error message when error prop is provided (standard mode)', () => {
      renderCombobox({ make: 'Maruti', error: 'Select or enter a model' })
      expect(screen.getByText('Select or enter a model')).toBeInTheDocument()
    })

    it('shows error message in custom mode too', () => {
      renderCombobox({ make: 'Maruti', isCustom: true, error: 'Select or enter a model' })
      expect(screen.getByText('Select or enter a model')).toBeInTheDocument()
    })
  })
})
